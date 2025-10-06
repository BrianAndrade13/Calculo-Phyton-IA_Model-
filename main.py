import os
import math
from datetime import datetime, date, timedelta
from typing import List, Optional, Tuple

from fastapi import FastAPI, HTTPException, Header, Depends, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# DB
from sqlalchemy import create_engine, Column, Integer, String, Float, Date, DateTime, text
from sqlalchemy.orm import declarative_base, sessionmaker, Session

# ==============================
# Config
# ==============================
MODEL_NAME = os.getenv("MODEL_NAME", "milagro-hourly-v1")
DEFAULT_UNIT = "°C"

# Coordenadas por defecto (Milagro, EC)
MILAGRO_LAT = -2.14882
MILAGRO_LON = -79.60273
DEFAULT_LAT = float(os.getenv("DEFAULT_LAT", str(MILAGRO_LAT)))
DEFAULT_LON = float(os.getenv("DEFAULT_LON", str(MILAGRO_LON)))

# ==============================
# App & CORS
# ==============================
app = FastAPI(title="Microservicio IA - Predicción Clima", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # endurecer en producción
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==============================
# DB setup (SQLite/Postgres)
# ==============================
Base = declarative_base()

DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./database.db")

# Solo SQLite necesita check_same_thread
is_sqlite = DATABASE_URL.startswith("sqlite")
connect_args = {"check_same_thread": False} if is_sqlite else {}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    pool_pre_ping=True,
    echo=False,  # pon True si quieres ver SQL
)

SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)

# --- Modelo
class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(String, index=True)         # tomado de x-token / x-user-id
    date = Column(Date, index=True)              # fecha solicitada (día)
    hour = Column(Integer, nullable=True)        # 0..23 (opcional)
    temp = Column(Float, nullable=False)
    at = Column(DateTime, nullable=False)

# Crea tablas según tus modelos
Base.metadata.create_all(bind=engine)

# --- Parche de esquema SOLO para SQLite (evita PRAGMA en Postgres)
def ensure_schema_sqlite_only():
    if not is_sqlite:
        return
    with engine.begin() as conn:
        cols = [row[1] for row in conn.execute(text("PRAGMA table_info(predictions)"))]
        if "hour" not in cols:
            conn.execute(text("ALTER TABLE predictions ADD COLUMN hour INTEGER"))

ensure_schema_sqlite_only()

def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# ==============================
# Utils / Fechas
# ==============================
def parse_day(day_str: str) -> date:
    try:
        return datetime.strptime(day_str, "%Y-%m-%d").date()
    except Exception:
        raise HTTPException(status_code=400, detail=f"Fecha inválida: {day_str}")

def _day_of_year(d: date) -> int:
    return d.timetuple().tm_yday

# ==============================
# Modelo físico–empírico (Milagro)
# ==============================
def _deg2rad(x: float) -> float: return x * math.pi / 180.0
def _rad2deg(x: float) -> float: return x * 180.0 / math.pi

def _solar_declination(doy: int) -> float:
    """Declinación solar (rad). Cooper (1969)."""
    return 0.409 * math.sin(2*math.pi*(doy - 81)/368)

def _sunset_hour_angle(lat_rad: float, dec: float) -> float:
    """Ángulo horario a la puesta (rad)."""
    x = -math.tan(lat_rad) * math.tan(dec)
    x = max(min(x, 1.0), -1.0)
    return math.acos(x)

def _approx_sun_times(d: date, lat_deg: float) -> Tuple[int, int, float]:
    """(hora_amanecer, hora_atardecer, duracion_dia_horas) aproximado."""
    doy = _day_of_year(d)
    dec = _solar_declination(doy)
    lat = _deg2rad(lat_deg)
    ws = _sunset_hour_angle(lat, dec)             # rad
    day_hours = 2 * _rad2deg(ws) / 15.0
    sunrise = int(round(12.0 - day_hours / 2.0))  # aprox LST
    sunset  = int(round(12.0 + day_hours / 2.0))
    # Rango razonable para Milagro
    sunrise = max(5, min(7, sunrise))
    sunset  = max(17, min(19, sunset))
    return sunrise, sunset, day_hours

def _seasonal_base(doy: int) -> Tuple[float, float]:
    """
    Tmedia y amplitud anual aproximadas para costa tropical.
    Ajusta si tienes histórico local.
    """
    tmean = 26.5 + 0.8 * math.sin(2*math.pi*(doy-40)/365.25)
    amp   = 4.0  + 0.6 * math.cos(2*math.pi*(doy-100)/365.25)
    return tmean, amp

def modelo_milagro_hourly(day_str: str, hour: Optional[int], lat: float, lon: float) -> float:
    """
    Modelo físico–empírico:
    - Estacional anual (Tmedia, amplitud)
    - Diurno por radiación (seno entre amanecer y ~15h; enfriamiento nocturno exponencial)
    - Soporta hour=0..23 (si None, promedio diario)
    """
    d = parse_day(day_str)
    doy = _day_of_year(d)
    tmean, amp = _seasonal_base(doy)

    # Estimar extremos del día:
    tmax = tmean + 0.7 * amp
    tmin = tmean - 0.7 * amp

    if hour is None:
        return round((tmax + tmin) / 2.0, 1)

    hour = max(0, min(23, hour))
    sunrise, sunset, _ = _approx_sun_times(d, lat)

    if sunrise < hour < 15:
        # Calentamiento: amanecer→15h (seno)
        span = (15 - sunrise)
        x = (hour - sunrise) / max(span, 1e-6)
        temp = tmin + (tmax - tmin) * math.sin(0.5 * math.pi * x)
    elif 15 <= hour <= sunset:
        # Enfriamiento suave tarde: 15h→atardecer (media seno invertida)
        span = (sunset - 15)
        x = (hour - 15) / max(span, 1e-6)
        temp = tmax - 0.5 * (tmax - tmean) * (1 - math.cos(math.pi * x))
    else:
        # Noche: enfriamiento exponencial (Linvill simplificado)
        if hour >= sunset:
            dt = hour - sunset
        else:
            dt = (24 - sunset) + hour
        k = 0.25  # 0.2–0.35 según nubosidad media
        temp = tmean - (tmean - tmin) * (1 - math.exp(-k * dt))

    # Micro-ajuste zonal leve
    temp += 0.02 * (lat - MILAGRO_LAT) + 0.005 * (lon - MILAGRO_LON)

    return round(temp, 1)

# ==============================
# Auth mínima (x-token -> user_id)
# ==============================
def require_user(
    x_token: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
) -> str:
    if x_user_id:  # prioridad al uid estable si te lo envía Flutter
        return x_user_id
    if not x_token:
        raise HTTPException(status_code=401, detail="Falta x-token")
    # intentar decodificar el JWT sin verificar firma para obtener uid/sub
    try:
        import jwt  # pip install PyJWT (opcional)
        payload = jwt.decode(x_token, options={"verify_signature": False})
        return str(payload.get("uid") or payload.get("sub") or x_token)
    except Exception:
        return x_token

# ==============================
# Schemas de salida (compatibles con tu Flutter)
# ==============================
class PredictOut(BaseModel):
    date: str
    temp: float
    unit: str = DEFAULT_UNIT

class SeriesPoint(BaseModel):
    date: str
    temp: float

class PredictRangeOut(BaseModel):
    unit: str = DEFAULT_UNIT
    series: List[SeriesPoint]

class HistoryItem(BaseModel):
    date: str
    temp: float
    at: str

class HistoryOut(BaseModel):
    unit: str = DEFAULT_UNIT
    items: List[HistoryItem]

# (Opcional) para 24 horas de un día
class HourPoint(BaseModel):
    hour: int
    temp: float

class DayHoursOut(BaseModel):
    date: str
    unit: str = DEFAULT_UNIT
    hours: List[HourPoint]

# ==============================
# Endpoints
# ==============================
@app.get("/api/health")
def health():
    return {"ok": True, "model": MODEL_NAME}

@app.get("/api/debug/mine")
def debug_mine(user_id: str = Depends(require_user), db: Session = Depends(get_db)):
    rows = db.query(Prediction).order_by(Prediction.id.desc()).all()
    return {
        "current_uid": (user_id or "")[:16],
        "count_all": len(rows),
        "items_all": [
            dict(id=r.id, uid=(r.user_id or "")[:16], date=r.date.isoformat(),
                 hour=r.hour, temp=r.temp, at=r.at.isoformat())
            for r in rows
        ],
        "items_mine": [
            dict(id=r.id, date=r.date.isoformat(), hour=r.hour,
                 temp=r.temp, at=r.at.isoformat())
            for r in rows if r.user_id == user_id
        ],
    }

@app.get("/api/predict", response_model=PredictOut)
def predict(
    date: str = Query(..., description="YYYY-MM-DD"),
    hour: int | None = Query(None, ge=0, le=23),
    lat: float = Query(DEFAULT_LAT),
    lon: float = Query(DEFAULT_LON),
    user_id: str = Depends(require_user),
    db: Session = Depends(get_db)
):
    temp = modelo_milagro_hourly(date, hour, lat, lon)

    rec = Prediction(
        user_id=user_id,
        date=parse_day(date),
        hour=hour,
        temp=float(temp),
        at=datetime.utcnow()
    )
    db.add(rec)
    db.commit()

    print("SAVE", (user_id or "")[:12], date, hour, temp)
    return PredictOut(date=date, temp=temp)

@app.get("/api/predict/range", response_model=PredictRangeOut)
def predict_range(
    start: str = Query(..., description="YYYY-MM-DD"),
    end: str = Query(..., description="YYYY-MM-DD"),
    lat: float = Query(DEFAULT_LAT),
    lon: float = Query(DEFAULT_LON),
    user_id: str = Depends(require_user)
):
    s = parse_day(start)
    e = parse_day(end)
    if e < s:
        raise HTTPException(status_code=400, detail="end < start")

    series: List[SeriesPoint] = []
    d = s
    while d <= e:
        ds = d.isoformat()
        series.append(SeriesPoint(date=ds, temp=modelo_milagro_hourly(ds, None, lat, lon)))
        d += timedelta(days=1)

    return PredictRangeOut(series=series)

@app.get("/api/predictions/mine", response_model=HistoryOut)
def predictions_mine(
    user_id: str = Depends(require_user),
    db: Session = Depends(get_db)
):
    rows = (
        db.query(Prediction)
          .filter(Prediction.user_id == user_id)
          .order_by(Prediction.at.desc())
          .all()
    )
    items = [
        HistoryItem(
            date=r.date.isoformat(),
            temp=float(r.temp),
            at=r.at.replace(microsecond=0).isoformat() + "Z"
        )
        for r in rows
    ]
    return HistoryOut(items=items)

@app.get("/api/predict/dayhours", response_model=DayHoursOut)
def predict_dayhours(
    date: str = Query(..., description="YYYY-MM-DD"),
    lat: float = Query(DEFAULT_LAT),
    lon: float = Query(DEFAULT_LON),
    user_id: str = Depends(require_user)
):
    d = parse_day(date)
    hours = [HourPoint(hour=h, temp=modelo_milagro_hourly(d.isoformat(), h, lat, lon)) for h in range(24)]
    return DayHoursOut(date=d.isoformat(), hours=hours)

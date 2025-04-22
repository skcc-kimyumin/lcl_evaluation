import redis
from core.config import get_setting

settings = get_setting()

connection_url = (
    f"rediss://:{settings.REDIS_ACCESS_KEY}@{settings.REDIS_HOST}:{settings.REDIS_PORT}/{settings.REDIS_DATABASE}?ssl_cert_reqs=required"
    if settings.REDIS_USE_SSL
    else f"redis://:{settings.REDIS_ACCESS_KEY}@{settings.REDIS_HOST}/{settings.REDIS_DATABASE}"
)
redis_pool = redis.ConnectionPool().from_url(connection_url, decode_responses=True)


def get_redis():
    with redis.StrictRedis(connection_pool=redis_pool) as r:
        yield r

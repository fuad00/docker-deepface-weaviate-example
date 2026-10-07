# docker-deepface-weaviate-example

Минимальный PoC связки **DeepFace** (распознавание лиц) + **Weaviate** (векторная БД): FastAPI-сервис принимает фото, снимает эмбеддинги через DeepFace и ищет/складывает их в Weaviate. Всё в одном `docker compose up`.

## Состав

- `fastapi/` — сервис на FastAPI (`main.py`, `models.py`): эндпоинты регистрации и поиска лиц
- `weaviate_db` — векторная БД (порт 8080 HTTP / 50051 gRPC), persistence в `./weaviate_data`
- Веса моделей DeepFace кешируются в `~/.deepface/weights` (маунтится в контейнер)

## Запуск

```bash
git clone https://github.com/fuad00/docker-deepface-weaviate-example
cd docker-deepface-weaviate-example

docker compose up
# API: http://<host>:8000  (Swagger: /docs)
# Weaviate: http://<host>:8080
```

## Дисклеймер

Пример архитектуры, не продакшен: anonymous-доступ к Weaviate включён, аутентификации нет.

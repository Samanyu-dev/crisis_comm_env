FROM node:20-alpine AS frontend-builder

WORKDIR /app/frontend

COPY frontend/package.json /app/frontend/package.json
RUN npm install

COPY frontend /app/frontend
RUN npm run build

FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY server/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r /app/requirements.txt

COPY . /app
COPY --from=frontend-builder /app/frontend/dist /app/frontend/dist

WORKDIR /app/server

EXPOSE 7860

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]

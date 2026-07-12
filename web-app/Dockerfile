FROM python:3.10-slim as backend

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy main project
COPY . /app/

# Install Python dependencies
RUN pip install --no-cache-dir -r web-app/backend/requirements.txt && \
    pip install --no-cache-dir -e .

WORKDIR /app/web-app/backend

EXPOSE 8000

CMD ["python", "main.py"]

---

FROM node:18-alpine as frontend

WORKDIR /app/web-app/frontend

# Copy frontend files
COPY web-app/frontend/ .

# Install dependencies
RUN npm install --legacy-peer-deps

# Build
RUN npm run build

FROM node:18-alpine

WORKDIR /app

# Install serve to serve the React build
RUN npm install -g serve

# Copy built frontend from previous stage
COPY --from=frontend /app/web-app/frontend/build /app/build

# Copy backend
COPY --from=backend /app /app

EXPOSE 3000 8000

# Run both services using a simple script
CMD ["sh", "-c", "cd /app/web-app/backend && python main.py &  serve -s /app/build -l 3000"]

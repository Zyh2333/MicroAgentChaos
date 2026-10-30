FROM python:3.10-slim

WORKDIR /app

# Install the proxy runtime dependencies.
RUN pip install --no-cache-dir "aiohttp==3.14.4" "PyYAML==6.0.3"

# Keep the package path for python -m proxy_pod.server.
COPY . /app/proxy_pod/

EXPOSE 18800

# The ConfigMap mounts the proxy configuration here.
ENTRYPOINT ["python", "-m", "proxy_pod.server"]
CMD ["--config", "/app/proxy_pod/config/faults.example.yaml", "--start-without-faults", "--listen", "0.0.0.0:18800"]

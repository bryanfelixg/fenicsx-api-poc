# Use the official FEniCSx/dolfinx image with PETSc/MPI preinstalled
FROM dolfinx/dolfinx:stable

# Keep things predictable on tiny machines
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=1 \
    NUMBA_NUM_THREADS=1

# FastAPI and plotting (dolfinx & petsc4py are already in the base image)
RUN python3 -m pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    pydantic \
    numpy \
    matplotlib

WORKDIR /app
COPY app /app/app

# Render provides $PORT; default to 8000 when running locally
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
# Gunakan image Python yang sesuai
FROM python:3.10-slim

# Set working directory di dalam container
WORKDIR /app

# Salin file requirements.txt dan install dependencies
COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

# Salin sisa file aplikasi ke dalam container
COPY . .

# Expose port yang digunakan aplikasi (misal: 5000 untuk Flask)
EXPOSE 5000

# Menjalankan aplikasi
CMD ["python", "app.py"]

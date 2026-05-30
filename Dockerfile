# استخدم صورة Python
FROM python:3.11-slim

# تثبيت نظام التبعيات المطلوبة
RUN apt-get update && apt-get install -y \
    curl \
    gnupg \
    unixodbc \
    unixodbc-dev \
    && rm -rf /var/lib/apt/lists/*

# تثبيت ODBC Driver 17 لـ SQL Server
RUN curl https://packages.microsoft.com/keys/microsoft.asc | apt-key add - \
    && curl https://packages.microsoft.com/config/debian/11/prod.list > /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y msodbcsql17 \
    && rm -rf /var/lib/apt/lists/*

# ضبط مجلد العمل
WORKDIR /app

# نسخ ملفات الاعتمادات وتثبيتها
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نسخ باقي الملفات
COPY . .

# تعريف متغير PORT
ENV PORT 8000

# تشغيل التطبيق
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}

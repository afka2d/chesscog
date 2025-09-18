#!/bin/bash

echo "🚀 Starting Local Development API..."
echo "📍 This will run on port 8001 (separate from production on port 8000)"
echo "🔧 Includes additional debugging features for development"
echo ""

# Check if port 8001 is already in use
if lsof -Pi :8001 -sTCP:LISTEN -t >/dev/null ; then
    echo "⚠️  Port 8001 is already in use. Stopping existing process..."
    pkill -f "python main_local_dev.py"
    sleep 2
fi

# Start the local development API
echo "Starting API on port 8001..."
python main_local_dev.py &

# Wait a moment for startup
sleep 5

# Test the API
echo ""
echo "🧪 Testing the local development API..."
python test_local_dev.py

echo ""
echo "✅ Local Development API is running!"
echo "📍 API URL: http://localhost:8001"
echo "🔧 Debug info: http://localhost:8001/debug/info"
echo "📋 Health check: http://localhost:8001/health"
echo ""
echo "To stop the API, run: pkill -f 'python main_local_dev.py'"

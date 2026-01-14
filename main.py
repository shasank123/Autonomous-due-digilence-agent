import uvicorn
import os
import sys

if __name__ == "__main__":
    # Add src to python path
    sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
    
    # Run the API application
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

# In src/api/main.py - Add persistence endpoints

@app.get("/analysis/{request_id}/state")
async def get_analysis_state(request_id: str):
    """Get current analysis state using persistence"""
    try:
        state = orchestrator.get_analysis_state(request_id)
        if state:
            return state
        else:
            raise HTTPException(status_code=404, detail="Analysis state not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/analysis/{request_id}/history") 
async def get_analysis_history(request_id: str):
    """Get analysis execution history"""
    try:
        history = orchestrator.get_analysis_history(request_id)
        return {"request_id": request_id, "history": history}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
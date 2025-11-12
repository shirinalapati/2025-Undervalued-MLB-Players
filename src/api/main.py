"""
FastAPI application for serving undervalued MLB hitter data.

Provides RESTful endpoints for programmatic access to:
- Player statistics (all 350 hitters with ≥200 PA from 2025 season)
- Undervalued player rankings based on UVS (Undervaluation Score)
- Available metrics and definitions

Note: The main project uses the interactive Plotly Dash dashboard (frontend/table_dashboard.py)
for most users. This API provides programmatic access for developers or integrations.

The API loads data from data/processed/comprehensive_stats_2025.csv.
"""

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
from pathlib import Path
import sys
import logging

sys.path.append(str(Path(__file__).parent.parent))

# Note: ML model removed - project uses UVS formula for ranking
# from src.models.undervalued_model import UndervaluedPlayerModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
MODEL_DIR = PROJECT_ROOT / "data" / "models"

# Initialize FastAPI app
app = FastAPI(
    title="Undervalued MLB Players API",
    description="API for identifying undervalued MLB players using advanced analytics",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify actual origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global data cache
_data_cache = {}
_model_cache = {}


def load_data(year: int = 2025):
    """Load and cache data for a given year."""
    if year not in _data_cache:
        data_path = DATA_PROCESSED / f"comprehensive_stats_{year}.csv"
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")
        _data_cache[year] = pd.read_csv(data_path)
        logger.info(f"Loaded data for {year}")
    return _data_cache[year]


# Note: ML model removed - API now uses UVS (Undervaluation Score) for ranking
# def load_model(year: int = 2025, target_type: str = 'hitter'):
#     """Load and cache model for a given year and type."""
#     ...


# Pydantic models for responses
class Player(BaseModel):
    """Player data model."""
    name: str
    player_id: Optional[str] = None
    position_type: str
    wOBA: Optional[float] = None
    xwOBA: Optional[float] = None
    ERA: Optional[float] = None
    xERA: Optional[float] = None
    value_score: Optional[float] = None
    undervalued_rank: Optional[int] = None
    performance_diff: Optional[float] = None
    
    class Config:
        from_attributes = True


class UndervaluedResponse(BaseModel):
    """Response model for undervalued players."""
    players: List[Player]
    year: int
    position_type: str
    count: int


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "Undervalued MLB Players API",
        "version": "1.0.0",
        "endpoints": {
            "/api/players": "Get all players",
            "/api/players/{player_id}": "Get specific player",
            "/api/undervalued": "Get undervalued player leaderboard",
            "/api/undervalued/hitters": "Get top undervalued hitters",
            "/api/undervalued/pitchers": "Get top undervalued pitchers",
            "/api/metrics": "Get available metrics"
        }
    }


@app.get("/api/players")
async def get_players(
    year: int = Query(2025, description="Season year"),
    position_type: Optional[str] = Query(None, description="Filter by position type (Hitter/Pitcher)"),
    limit: int = Query(100, description="Maximum number of results")
):
    """Get all players with their statistics."""
    try:
        df = load_data(year)
        
        if position_type:
            df = df[df.get('position_type', '') == position_type]
        
        # Convert to dict format
        players = df.head(limit).to_dict('records')
        
        return {
            "players": players,
            "count": len(players),
            "year": year
        }
    except Exception as e:
        logger.error(f"Error getting players: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/players/{player_id}")
async def get_player(player_id: str, year: int = Query(2025, description="Season year")):
    """Get specific player by ID."""
    try:
        df = load_data(year)
        
        # Try to find by player_id or name
        player = df[
            (df.get('player_id', '') == player_id) |
            (df.get('Name', '') == player_id) |
            (df.get('name', '') == player_id)
        ]
        
        if player.empty:
            raise HTTPException(status_code=404, detail="Player not found")
        
        return player.iloc[0].to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting player: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/undervalued", response_model=UndervaluedResponse)
async def get_undervalued_players(
    year: int = Query(2025, description="Season year"),
    position_type: Optional[str] = Query(None, description="Filter by position type"),
    top_n: int = Query(20, description="Number of top players to return")
):
    """Get undervalued player leaderboard."""
    try:
        df = load_data(year)
        
        if position_type:
            df = df[df.get('position_type', '') == position_type]
            target_type = 'hitter' if position_type == 'Hitter' else 'pitcher'
        else:
            # Default to hitters if no filter
            df = df[df.get('position_type', '') == 'Hitter']
            target_type = 'hitter'
        
        if len(df) == 0:
            raise HTTPException(status_code=404, detail="No players found")
        
        # Use UVS (Undervaluation Score) for ranking instead of ML model
        if 'uvs' in df.columns:
            df = df.sort_values('uvs', ascending=False)
            undervalued = df.head(top_n)
        elif 'undervalued_rank' in df.columns:
            df = df.sort_values('undervalued_rank', ascending=True)
            undervalued = df.head(top_n)
        else:
            # Fallback: sort by WAR per salary if available
            if 'war_per_salary' in df.columns:
                df = df.sort_values('war_per_salary', ascending=False)
                undervalued = df.head(top_n)
            else:
                undervalued = df.head(top_n)
        
        # Convert to response format
        players = []
        for _, row in undervalued.iterrows():
            players.append(Player(
                name=row.get('Name', row.get('name', 'Unknown')),
                player_id=str(row.get('player_id', '')),
                position_type=row.get('position_type', ''),
                wOBA=row.get('wOBA'),
                xwOBA=row.get('xwOBA'),
                ERA=row.get('ERA'),
                xERA=row.get('xERA'),
                value_score=row.get('value_score'),
                undervalued_rank=int(row.get('undervalued_rank', 0)),
                performance_diff=row.get('performance_diff')
            ))
        
        return UndervaluedResponse(
            players=players,
            year=year,
            position_type=position_type or target_type,
            count=len(players)
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting undervalued players: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/undervalued/hitters")
async def get_undervalued_hitters(
    year: int = Query(2025, description="Season year"),
    top_n: int = Query(20, description="Number of top players to return")
):
    """Get top undervalued hitters."""
    return await get_undervalued_players(year=year, position_type='Hitter', top_n=top_n)


@app.get("/api/undervalued/pitchers")
async def get_undervalued_pitchers(
    year: int = Query(2025, description="Season year"),
    top_n: int = Query(20, description="Number of top players to return")
):
    """Get top undervalued pitchers."""
    return await get_undervalued_players(year=year, position_type='Pitcher', top_n=top_n)


@app.get("/api/metrics")
async def get_metrics():
    """Get information about available metrics."""
    return {
        "metrics": {
            "wOBA": {
                "name": "Weighted On-Base Average",
                "description": "Measures overall offensive value, weighting each outcome by its run value",
                "range": "0.000 - 1.000+"
            },
            "xwOBA": {
                "name": "Expected Weighted On-Base Average",
                "description": "Expected wOBA based on quality of contact and plate discipline",
                "range": "0.000 - 1.000+"
            },
            "ERA": {
                "name": "Earned Run Average",
                "description": "Average earned runs allowed per 9 innings",
                "range": "0.00 - 10.00+"
            },
            "xERA": {
                "name": "Expected Earned Run Average",
                "description": "Expected ERA based on quality of contact allowed",
                "range": "0.00 - 10.00+"
            },
            "value_score": {
                "name": "Value Score",
                "description": "Composite score combining performance over expectation and salary efficiency",
                "range": "0.0 - 1.0"
            },
            "performance_diff": {
                "name": "Performance Difference",
                "description": "Actual performance minus expected performance (positive = outperforming)",
                "range": "Negative to Positive"
            }
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


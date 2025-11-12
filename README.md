# Finding Undervalued MLB Players (2025 Season)

A comprehensive baseball analytics project that identifies undervalued hitters by analyzing advanced statistics, expected performance metrics, and salary efficiency. This project demonstrates skills in data engineering, statistical analysis, and data visualization.

## Project Overview

This project implements a "Moneyball" approach to identify MLB hitters who are undervalued based on their underlying performance metrics, contact quality, plate discipline, and salary efficiency. The system analyzes **all the 350 hitters from the 2025 MLB regular season** who had at least 200 plate appearances.

The project features an **interactive web dashboard** that displays comprehensive advanced statistics and ranks players using the **Undervaluation Score (UVS)** formula, which combines 7 weighted indexes to identify players whose expected performance exceeds their actual results and salary.

## Features

- **Comprehensive Data Pipeline**: Automated collection of 2025 MLB season data from Statcast and FanGraphs via pybaseball
- **Advanced Statistics**: Tracks 30+ advanced metrics including:
  - Expected Performance: xwOBA, xSLG, xBA, xISO
  - Contact Quality: Barrel%, HardHit%, Exit Velocity, Sweet Spot%
  - Plate Discipline: BB%, K%, Chase%, Z-Contact%, Contact%
  - Batted Ball Profile: GB%, FB%, LD%, Pull%, Oppo%
  - Run Production: wRC+, wOBA, OPS, ISO, R, RBI
  - Value Metrics: WAR, WAR per $1M, Salary
- **Undervaluation Score (UVS)**: Custom composite metric that identifies undervalued players by combining:
  - Expected Performance Index (25%)
  - Contact Quality Index (20%)
  - Plate Discipline Index (15%)
  - Run Production Index (15%)
  - Salary Efficiency (10%)
  - Luck Adjustment (10%)
- **Interactive Dashboard**: Plotly Dash web application with:
  - **About This Page**: Project overview and methodology explanation
  - **All Players Stats**: Comprehensive table with all 350 hitters and their statistics
  - **Undervalued Players**: UVS formula breakdown and ranked player list
  - Sortable, filterable tables with CSV export functionality

## Tech Stack

- **Python 3.9+**: Core programming language
- **Pandas**: Data manipulation and analysis
- **Plotly Dash**: Interactive web dashboard
- **pybaseball**: MLB data collection (Statcast, FanGraphs)
- **Dash Bootstrap Components**: Professional UI styling
- **NumPy**: Numerical computations for z-score calculations

## Project Structure

```
Undervalued_MLBPlayers/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/              # Raw data files
│   ├── processed/        # Cleaned and processed data
│   └── models/           # Model files (if any)
├── src/
│   ├── data_pipeline/    # Data collection and ETL
│   │   ├── fetch_advanced_metrics.py
│   │   ├── fetch_comprehensive.py
│   │   ├── fetch_salary_data.py
│   │   └── fetch_statcast_pitch_data.py
│   ├── models/           # Statistical models (currently empty)
│   ├── api/              # FastAPI application (optional)
│   │   └── main.py
│   └── utils/            # Utility functions
│       ├── metrics.py
│       ├── tova_metrics.py
│       ├── uvs_metrics.py
│       └── additional_metrics.py
├── frontend/
│   └── table_dashboard.py  # Main dashboard application
├── notebooks/            # Jupyter notebooks for analysis
└── find_undervalued.py   # Main script to run pipeline
```

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd Undervalued_MLBPlayers
```

2. Create a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Data Collection

Run the comprehensive data pipeline to collect 2025 MLB season data:

```bash
python find_undervalued.py
```

Or run the data pipeline directly:

```bash
python src/data_pipeline/fetch_comprehensive.py
```

This will:
- Fetch batting statistics for all hitters with ≥200 PA
- Collect Statcast metrics (Barrel%, HardHit%, Exit Velocity, etc.)
- Merge salary data
- Calculate all advanced metrics and composite scores
- Save results to `data/processed/comprehensive_stats_2025.csv`

### 2. Launch the Dashboard

Activate your virtual environment and start the dashboard:

```bash
source venv/bin/activate  # On Windows: venv\Scripts\activate
python frontend/table_dashboard.py
```

The dashboard will be available at: `http://localhost:8050`

**Note:** If port 8050 is already in use, you may need to kill the existing process first:
```bash
lsof -ti:8050 | xargs kill -9  # On Windows, use Task Manager or: netstat -ano | findstr :8050
```

### Dashboard Features

The dashboard includes three tabs:

1. **About This Page** (Default): 
   - Explains the project methodology
   - Describes the UVS formula and its components
   - Overview of the website's purpose

2. **All Players Stats**:
   - Comprehensive table of all 350 hitters
   - 30+ statistics per player
   - Sortable and filterable columns
   - CSV export functionality
   - Default sorting by UVS rank

3. **Undervalued Players**:
   - Detailed breakdown of the UVS formula
   - Z-score calculation explanation
   - Complete ranked list of all players by UVS score
   - Sortable and filterable table

## Key Metrics Analyzed

### Expected Performance
- **xwOBA**: Expected Weighted On-Base Average
- **xSLG**: Expected Slugging Percentage
- **xBA**: Expected Batting Average
- **xISO**: Expected Isolated Power (xSLG - xBA)

### Contact Quality & Power
- **Barrel%**: Percentage of batted balls hit with ideal exit velocity + launch angle
- **HardHit%**: Percentage of batted balls hit ≥ 95 mph
- **Exit Velocity**: Average speed (mph) of all batted balls
- **Sweet Spot%**: Percentage of batted balls with launch angle 8–32°

### Plate Discipline
- **BB%**: Walks / Plate Appearances
- **K%**: Strikeouts / Plate Appearances
- **Chase% (O-Swing%)**: Percentage of swings at pitches outside strike zone
- **Z-Contact%**: Percentage of swings on in-zone pitches that make contact
- **Contact%**: Percentage of all swings that make contact

### Batted Ball Profile
- **GB%**: Percentage of batted balls that are grounders
- **FB%**: Percentage that are fly balls
- **LD%**: Percentage that are line drives
- **Pull%**: Percentage of balls hit to pull side
- **Oppo%**: Percentage hit to opposite field

### Run Production & Value
- **wRC+**: Weighted Runs Created Plus (100 = league average)
- **WAR**: Wins Above Replacement
- **WAR/$1M**: WAR divided by salary (in millions)
- **Salary ($M)**: 2025 salary in millions

## Undervaluation Score (UVS) Methodology

The UVS formula identifies undervalued players by combining 7 weighted indexes:

### Formula Components

**UVS = 0.25(EPI) + 0.20(CQI) + 0.15(PDI) + 0.15(RPI) + 0.10(SE) + 0.10(LA)**

Where:

1. **EPI (Expected Performance Index)** - 25% weight
   - Mean of z-scores: xwOBA, xSLG, xBA, xISO
   - Identifies players with strong underlying performance

2. **CQI (Contact Quality Index)** - 20% weight
   - Mean of z-scores: Barrel%, HardHit%, Exit Velocity, Sweet Spot%
   - Measures raw hitting talent and power potential

3. **PDI (Plate Discipline Index)** - 15% weight
   - z(BB%) – z(K%) – z(O-Swing%) + z(Z-Contact%) + z(Contact%)
   - Strong predictor of future performance

4. **RPI (Run Production Index)** - 15% weight
   - Mean of z-scores: wRC+, wOBA, OPS, ISO, R, RBI
   - Measures actual offensive contribution

5. **SE (Salary Efficiency)** - 10% weight
   - z(WAR per $1M)
   - Identifies value per dollar spent

6. **LA (Luck Adjustment)** - 10% weight
   - Mean of z-scores: (xwOBA - wOBA), (xBA - BA), (xSLG - SLG)
   - Identifies players who have been unlucky

All components use z-scores: `z = (x - x̄) / σ`

### Interpretation

- **Higher UVS = More Undervalued**: Players with high UVS have strong expected metrics, contact quality, and efficiency relative to their salary and actual results
- **Positive Luck Adjustment**: Indicates a player has been unlucky (expected > actual)
- **High Salary Efficiency**: Player provides exceptional value per dollar

## Data Sources

- **Statcast Data**: Via pybaseball (expected stats, contact quality metrics)
- **FanGraphs Data**: Via pybaseball (traditional and advanced statistics)
- **Salary Data**: 2025 MLB player salaries (from Sportrac database)

## Contributing

This is a portfolio project for demonstrating baseball analytics and software engineering skills.


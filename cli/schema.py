from typing import List, Optional, Dict, Union
from pydantic import BaseModel, Field
from datetime import date

# --- 1. Executive Summary ---

class ThesisPillar(BaseModel):
    pillar: str = Field(..., description="Short title of the pillar")
    description: str = Field(..., description="One sentence description")

class RiskRewardScenario(BaseModel):
    scenario: str = Field(..., description="Bull, Base, or Bear")
    implied_value: str = Field(..., description="Implied Value / Spread")
    return_vs_current: str = Field(..., description="Return vs Current %")
    drivers: str = Field(..., description="High-Level Drivers")

class ExecutiveSummary(BaseModel):
    investment_summary: str = Field(..., description="4-6 sentences answering What, Idea, Why Now, Risk/Reward")
    thesis_pillars: List[ThesisPillar] = Field(..., min_items=3, max_items=5)
    variant_view: str = Field(..., description="Consensus vs Our View")
    risk_reward_table: List[RiskRewardScenario]

# --- 2. Review Period Overview ---

class PriceStats(BaseModel):
    period: str
    return_pct: float
    volatility: float
    max_drawdown: float

class FundamentalMetric(BaseModel):
    metric: str
    start_value: float
    end_value: float
    change_pct: float
    comment: str

class Event(BaseModel):
    date: str
    event_type: str
    description: str
    impact: str

class ReviewPeriodOverview(BaseModel):
    price_stats: PriceStats
    fundamental_evolution: List[FundamentalMetric]
    event_timeline: List[Event]
    attribution_narrative: str

# --- 3. Business & Strategic Position ---

class Segment(BaseModel):
    name: str
    revenue_mix: float
    profit_mix: float
    growth: float

class BusinessOverview(BaseModel):
    description: str
    segments: List[Segment]
    revenue_drivers: str

class CompetitiveContext(BaseModel):
    competitors: List[str]
    market_structure: str
    moat_analysis: str
    structural_trends: str

class CapitalAllocation(BaseModel):
    capital_structure: str
    allocation_track_record: str
    governance_issues: Optional[str]

class BusinessStrategy(BaseModel):
    overview: BusinessOverview
    competitive_context: CompetitiveContext
    capital_allocation: CapitalAllocation

# --- 4. Thesis & Drivers ---

class DetailedThesisPillar(BaseModel):
    title: str
    claim: str
    evidence_analysis: str
    mechanism: str
    market_vs_view: str

class ThesisDrivers(BaseModel):
    pillars: List[DetailedThesisPillar]

# --- 5. Valuation & Scenarios ---

class ValuationMetric(BaseModel):
    metric: str
    value: float
    peer_avg: float
    history_avg: float

class Scenario(BaseModel):
    name: str # Bull, Base, Bear
    revenue_growth: str
    ebitda_margin: str
    eps_fcf: str
    implied_valuation: float
    upside_downside: float
    probability: float

class ValuationScenarios(BaseModel):
    current_valuation: List[ValuationMetric]
    valuation_frameworks: str # Text description of DCF/Relative
    scenario_analysis: List[Scenario]
    interpretation: str

# --- 6. Risks & Falsification ---

class Risk(BaseModel):
    category: str # Fundamental, Structural, etc.
    description: str
    impact: str

class FalsificationCondition(BaseModel):
    condition: str
    threshold: str

class RisksFalsification(BaseModel):
    fundamental_risks: List[Risk]
    structural_risks: List[Risk]
    falsification_conditions: List[FalsificationCondition]

# --- 7. Catalysts & Monitoring ---

class Catalyst(BaseModel):
    event: str
    date_window: str
    expected_impact: str
    what_to_watch: str

class MonitoringPlan(BaseModel):
    kpis: List[str]
    data_sources: List[str]
    frequency: str
    triggers: str

class CatalystsMonitoring(BaseModel):
    upcoming_catalysts: List[Catalyst]
    monitoring_plan: MonitoringPlan

# --- 8. Alternative Views ---

class AlternativeView(BaseModel):
    case_name: str # Bull Case / Bear Case
    summary: str
    key_assumptions: str
    support_from_review: str

class AlternativeViews(BaseModel):
    bull_case: AlternativeView
    bear_case: AlternativeView
    major_unknowns: List[str]

# --- Root Object ---

class FrontMatter(BaseModel):
    asset: str
    ticker: str
    company_name: str
    recommendation: str
    target_price: str
    current_price: str
    upside_downside: str
    region_sector: str
    as_of_date: str
    review_period: str
    prepared_by: str = "TradingAgents AI"
    intended_use: str
    illustrative_view: str # Long / Short / Neutral
    confidence: str # Low / Medium / High

class InvestmentMemo(BaseModel):
    front_matter: FrontMatter
    executive_summary: ExecutiveSummary
    review_period: ReviewPeriodOverview
    business_strategy: BusinessStrategy
    thesis: ThesisDrivers
    valuation: ValuationScenarios
    risks: RisksFalsification
    catalysts: CatalystsMonitoring
    alternatives: AlternativeViews
    data_sources: List[str]

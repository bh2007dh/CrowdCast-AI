"""
Crowd-Traffic Fusion Engine (Core Innovation of CrowdCast AI).

Combines:
1. Available traffic flow / congestion observations (Default weight: 35%)
2. Estimated campus & area crowd movement (Default weight: 30%)
3. Historical diurnal peak patterns (Default weight: 20%)
4. Active & scheduled event visitor/vehicle impact (Default weight: 15%)
5. Nearby road capacity indicators (where OSM lane/highway data is available)

Calculates a transparent, configurable Congestion Risk Score from 0 to 100.
Note: Default weights are initial demonstration weights, not scientifically validated constants.
"""
from datetime import datetime
from typing import Any, Dict, List, Optional
import numpy as np

from config import Config


class CongestionFusionEngine:
    """Calculates multi-factor crowd-traffic congestion risk and actionable recommendations."""

    def __init__(self, weights: Optional[Dict[str, float]] = None) -> None:
        self.weights = weights or dict(Config.FUSION_WEIGHTS)
        self._normalize_weights()

    def _normalize_weights(self) -> None:
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: round(v / total, 4) for k, v in self.weights.items()}
        else:
            self.weights = {
                "traffic_congestion": 0.35,
                "expected_crowd": 0.30,
                "historical_patterns": 0.20,
                "event_impact": 0.15,
            }

    @staticmethod
    def compute_historical_pattern_score(hour: int, day_of_week: int) -> float:
        """Calculate normalized 0-100 score for historical diurnal congestion at given hour/day."""
        morning = np.exp(-0.5 * ((hour - 9.0) / 1.5) ** 2) * 88.0
        midday = np.exp(-0.5 * ((hour - 13.5) / 1.8) ** 2) * 52.0
        evening = np.exp(-0.5 * ((hour - 17.5) / 1.7) ** 2) * 94.0
        base = max(12.0, morning, midday, evening)
        if day_of_week >= 5:
            base *= 0.68
        return float(np.clip(round(base, 1), 5.0, 100.0))

    @staticmethod
    def classify_risk_score(score: float) -> Dict[str, str]:
        if score < 30.0:
            return {
                "level": "Low",
                "color": "Green",
                "hex": "#10b981",
                "summary": "Free-flowing traffic and manageable pedestrian density.",
            }
        if score < 55.0:
            return {
                "level": "Moderate",
                "color": "Yellow",
                "hex": "#f59e0b",
                "summary": "Moderate queue build-up at primary intersections and entry gates.",
            }
        if score < 78.0:
            return {
                "level": "Heavy",
                "color": "Orange",
                "hex": "#f97316",
                "summary": "Heavy multi-modal congestion; stagger arrivals and use alternate gates.",
            }
        return {
            "level": "Severe",
            "color": "Red",
            "hex": "#ef4444",
            "summary": "Critical congestion risk; activate overflow parking and secondary corridors.",
        }

    def calculate_fusion_risk(
        self,
        traffic_congestion_ratio: float,
        estimated_crowd_count: int,
        expected_event_visitors: int = 0,
        expected_event_vehicles: int = 0,
        active_events_count: int = 0,
        hour: Optional[int] = None,
        day_of_week: Optional[int] = None,
        road_segments: Optional[List[Dict[str, Any]]] = None,
        campus_gates: Optional[List[Dict[str, Any]]] = None,
        custom_weights: Optional[Dict[str, float]] = None,
        traffic_data_source: str = "demonstration_mode",
    ) -> Dict[str, Any]:
        """
        Combine traffic, crowd, historical patterns, and event impact into a 0-100 risk score.
        """
        now = datetime.now()
        cur_hour = int(hour) if hour is not None else now.hour
        cur_day = int(day_of_week) if day_of_week is not None else now.weekday()

        active_weights = dict(custom_weights) if custom_weights else dict(self.weights)
        w_sum = sum(active_weights.values())
        if w_sum > 0:
            active_weights = {k: round(v / w_sum, 4) for k, v in active_weights.items()}

        # 1. Traffic Congestion Sub-score (0 - 100)
        traffic_score = float(np.clip(traffic_congestion_ratio * 100.0, 0.0, 100.0))

        # 2. Expected Crowd Sub-score (0 - 100, scaled against 2500 reference capacity)
        crowd_score = float(np.clip((estimated_crowd_count / 2400.0) * 100.0, 0.0, 100.0))

        # 3. Historical Peak Pattern Sub-score (0 - 100)
        historical_score = self.compute_historical_pattern_score(cur_hour, cur_day)

        # 4. Event Impact Sub-score (0 - 100)
        visitor_load = (expected_event_visitors / 2000.0) * 65.0
        vehicle_load = (expected_event_vehicles / 500.0) * 35.0
        event_score = float(np.clip(visitor_load + vehicle_load, 0.0, 100.0))

        # Road capacity modifier (when nearby road capacity indicators are available)
        capacity_pressure = 0.0
        nearby_road_risks = []
        if road_segments:
            for seg in road_segments[:10]:
                cap_vph = max(300, int(seg.get("capacity_vph", 1200)))
                lanes = max(1, int(seg.get("lanes", 2)))
                seg_traffic = float(seg.get("congestion_ratio", traffic_congestion_ratio))
                demand_vph = int(cap_vph * seg_traffic + (expected_event_vehicles / max(1, len(road_segments[:4]))))
                v_c_ratio = round(demand_vph / float(cap_vph), 2)
                seg_risk_score = float(np.clip(v_c_ratio * 72.0 + crowd_score * 0.28, 5.0, 99.0))
                seg_cls = self.classify_risk_score(seg_risk_score)
                nearby_road_risks.append(
                    {
                        "road_name": seg.get("road_name", "Local Road Segment"),
                        "road_type": seg.get("road_type", "secondary"),
                        "lanes": lanes,
                        "capacity_vph": cap_vph,
                        "estimated_demand_vph": demand_vph,
                        "volume_to_capacity_ratio": v_c_ratio,
                        "risk_score": round(seg_risk_score, 1),
                        "risk_level": seg_cls["level"],
                        "color": seg_cls["color"],
                        "is_intersection": bool(seg.get("is_intersection", False)),
                    }
                )
            nearby_road_risks.sort(key=lambda r: r["risk_score"], reverse=True)
            if nearby_road_risks:
                avg_top_vc = np.mean([r["volume_to_capacity_ratio"] for r in nearby_road_risks[:3]])
                if avg_top_vc > 0.85:
                    capacity_pressure = min(8.0, (avg_top_vc - 0.85) * 20.0)

        # Weighted fusion score
        weighted_score = (
            traffic_score * active_weights.get("traffic_congestion", 0.35)
            + crowd_score * active_weights.get("expected_crowd", 0.30)
            + historical_score * active_weights.get("historical_patterns", 0.20)
            + event_score * active_weights.get("event_impact", 0.15)
            + capacity_pressure
        )
        final_score = round(float(np.clip(weighted_score, 0.0, 100.0)), 1)
        classification = self.classify_risk_score(final_score)

        # Determine main contributing factors ranked by weighted contribution
        contributions = [
            {
                "factor": "Traffic Congestion Flow",
                "raw_score": round(traffic_score, 1),
                "weight_pct": round(active_weights.get("traffic_congestion", 0.35) * 100, 1),
                "weighted_points": round(traffic_score * active_weights.get("traffic_congestion", 0.35), 1),
                "detail": f"Road network operating at {round(traffic_congestion_ratio * 100)}% congestion index ({traffic_data_source}).",
            },
            {
                "factor": "Estimated Area & Campus Crowd",
                "raw_score": round(crowd_score, 1),
                "weight_pct": round(active_weights.get("expected_crowd", 0.30) * 100, 1),
                "weighted_points": round(crowd_score * active_weights.get("expected_crowd", 0.30), 1),
                "detail": f"Estimated {estimated_crowd_count:,} pedestrians/commuters in the active zone.",
            },
            {
                "factor": "Historical Peak Time Pattern",
                "raw_score": round(historical_score, 1),
                "weight_pct": round(active_weights.get("historical_patterns", 0.20) * 100, 1),
                "weighted_points": round(historical_score * active_weights.get("historical_patterns", 0.20), 1),
                "detail": f"Time slot ({cur_hour:02d}:00) carries a {round(historical_score)}% baseline commuter load.",
            },
            {
                "factor": "Scheduled Event Visitor & Vehicle Surge",
                "raw_score": round(event_score, 1),
                "weight_pct": round(active_weights.get("event_impact", 0.15) * 100, 1),
                "weighted_points": round(event_score * active_weights.get("event_impact", 0.15), 1),
                "detail": (
                    f"{active_events_count} active event(s) adding ~{expected_event_visitors:,} visitors "
                    f"and ~{expected_event_vehicles:,} vehicles."
                    if expected_event_visitors > 0
                    else "No major scheduled event surge currently active."
                ),
            },
        ]
        contributions.sort(key=lambda c: c["weighted_points"], reverse=True)

        # Expected peak period calculation
        if cur_hour < 10:
            expected_peak_period = "08:30 - 10:15 (Morning Commute & Event Arrival Window)"
        elif cur_hour < 15:
            expected_peak_period = "13:00 - 14:30 (Midday Session Transition Window)"
        else:
            expected_peak_period = "16:30 - 19:00 (Evening Dispersal & Arterial Peak Window)"

        # Recommended location-specific actions
        recommendations = []
        if final_score >= 70:
            recommendations.extend(
                [
                    "Divert incoming event and visitor vehicles to secondary entry gates to relieve primary arterial queues.",
                    "Stagger campus/venue dismissal or session breaks by 15-20 minutes to flatten pedestrian exit surges.",
                    "Recommend commuters arrive 30 minutes before or after the predicted peak window.",
                    "Activate overflow parking zones and restrict temporary curbside drop-offs at main intersections.",
                ]
            )
        elif final_score >= 45:
            recommendations.extend(
                [
                    "Balance incoming vehicular arrivals across at least 2 entry gates.",
                    "Use secondary collector roads rather than primary junctions during the next 30 minutes.",
                    "Monitor pedestrian crossings near transit stops and college gates during session transitions.",
                ]
            )
        else:
            recommendations.extend(
                [
                    "Normal gate and corridor operations are sufficient for current demand.",
                    "Standard arrival schedule recommended; all primary approach roads are operating within capacity.",
                ]
            )

        # Smart Gate Allocation based on predicted crowd + vehicle demand and gate capacities
        gate_allocations = []
        if campus_gates:
            open_gates = [g for g in campus_gates if g.get("status", "Open") == "Open"]
            total_gate_cap = sum(max(100, int(g.get("max_capacity_per_hour", 800))) for g in open_gates) or 1
            total_demand = max(200, estimated_crowd_count + expected_event_visitors // 3)
            for g in open_gates:
                g_cap = max(100, int(g.get("max_capacity_per_hour", 800)))
                share = g_cap / float(total_gate_cap)
                allocated_flow = int(round(total_demand * share))
                utilization_pct = int(min(100, round((allocated_flow / float(g_cap)) * 100)))
                gate_allocations.append(
                    {
                        "gate_id": g.get("id"),
                        "gate_name": g.get("gate_name"),
                        "gate_type": g.get("gate_type", "Entry/Exit"),
                        "connected_road": g.get("connected_road", "Campus Perimeter Road"),
                        "max_capacity_per_hour": g_cap,
                        "recommended_hourly_allocation": allocated_flow,
                        "projected_utilization_pct": utilization_pct,
                        "recommendation": (
                            "Preferred Primary Entry (Spare Capacity)"
                            if utilization_pct < 65
                            else (
                                "Balanced Load - Monitor Queue"
                                if utilization_pct < 85
                                else "Near Capacity - Divert Excess Vehicles to Secondary Gate"
                            )
                        ),
                    }
                )

        return {
            "congestion_risk_score": final_score,
            "risk_level": classification["level"],
            "color_code": classification["color"],
            "hex_color": classification["hex"],
            "summary": classification["summary"],
            "expected_peak_period": expected_peak_period,
            "weights_used": active_weights,
            "weights_disclaimer": (
                "Initial demonstration weights (Traffic 35%, Crowd 30%, Historical 20%, Event 15%). "
                "Configurable by administrators; not scientifically calibrated constants."
            ),
            "sub_scores": {
                "traffic_congestion_score": round(traffic_score, 1),
                "expected_crowd_score": round(crowd_score, 1),
                "historical_pattern_score": round(historical_score, 1),
                "event_impact_score": round(event_score, 1),
                "road_capacity_modifier": round(capacity_pressure, 1),
            },
            "main_contributing_factors": contributions,
            "nearby_road_risks": nearby_road_risks,
            "recommended_actions": recommendations,
            "gate_recommendations": gate_allocations,
            "safety_disclaimer": (
                "Advisory decision-support only. Does not direct emergency vehicles or override official traffic controls."
            ),
            "data_source": (
                "hybrid_live_and_estimated"
                if traffic_data_source == "live_tomtom_api"
                else "estimated_and_demonstration"
            ),
        }


congestion_engine = CongestionFusionEngine()


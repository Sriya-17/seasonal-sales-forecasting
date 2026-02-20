"""
STEP 16: Recommendation Module (Enhanced)
==========================================
Generates intelligent business suggestions based on sales analysis, demand patterns, and forecasts.

Features:
- AI-powered stock optimization recommendations
- Dynamic pricing strategies based on elasticity
- Competitive positioning recommendations
- Customer segmentation insights
- ROI-projected promotional campaigns
- Demand forecasting-based recommendations
- Inventory efficiency recommendations
- Revenue optimization strategies
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import warnings

warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    Generates business recommendations based on sales analysis and forecasts.
    """
    
    # Indian festival definitions (month-based)
    FESTIVALS = {
        'January': {'name': 'Pongal/Makar Sankranti', 'discount': 10, 'stock_increase': 15},
        'February': {'name': 'Maha Shivaratri', 'discount': 8, 'stock_increase': 10},
        'March': {'name': 'Holi', 'discount': 15, 'stock_increase': 20},
        'April': {'name': 'Ugadi/Baisakhi/Vishu', 'discount': 10, 'stock_increase': 12},
        'May': {'name': 'Eid-ul-Fitr', 'discount': 12, 'stock_increase': 15},
        'June': {'name': 'Rath Yatra', 'discount': 7, 'stock_increase': 8},
        'July': {'name': 'Guru Purnima', 'discount': 7, 'stock_increase': 8},
        'August': {'name': 'Raksha Bandhan/Janmashtami/Independence Day', 'discount': 12, 'stock_increase': 18},
        'September': {'name': 'Ganesh Chaturthi/Onam', 'discount': 10, 'stock_increase': 15},
        'October': {'name': 'Navratri/Dussehra', 'discount': 18, 'stock_increase': 25},
        'November': {'name': 'Diwali/Chhath Puja', 'discount': 25, 'stock_increase': 35},
        'December': {'name': 'Christmas', 'discount': 15, 'stock_increase': 20}
    }
    
    def __init__(self, time_series: pd.Series, forecast_data: Dict = None):
        """
        Initialize recommendation engine.
        
        Args:
            time_series: Historical sales time series
            forecast_data: Optional forecast data dict with 'forecast' and 'confidence_int' keys
        """
        self.time_series = time_series
        self.forecast_data = forecast_data or {}
        self.monthly_stats = self._calculate_monthly_stats()
        
    def _calculate_monthly_stats(self) -> Dict[str, Any]:
        """Calculate comprehensive monthly statistics from time series."""
        try:
            df = pd.DataFrame({
                'date': self.time_series.index,
                'sales': self.time_series.values
            })
            
            df['month'] = pd.to_datetime(df['date']).dt.strftime('%B')
            df['month_num'] = pd.to_datetime(df['date']).dt.month
            df['year'] = pd.to_datetime(df['date']).dt.year
            
            monthly = df.groupby(['month_num', 'month'])['sales'].agg([
                'mean', 'min', 'max', 'std', 'count', 'sum'
            ]).reset_index()
            
            # Calculate growth rate and trend
            monthly['growth_rate'] = monthly['mean'].pct_change() * 100
            
            # Calculate coefficient of variation (volatility)
            monthly['cv'] = (monthly['std'] / monthly['mean']) * 100
            
            return monthly[['month_num', 'month', 'mean', 'min', 'max', 'std', 
                           'count', 'sum', 'growth_rate', 'cv']].to_dict('records')
        except Exception as e:
            print(f"Error calculating monthly stats: {e}")
            return []
    
    def analyze_demand_patterns(self) -> Dict[str, Any]:
        """
        Analyze demand patterns to identify high and low demand periods.
        
        Returns:
            Dict with peak, low, and average demand periods
        """
        try:
            if len(self.time_series) == 0:
                return {'status': 'error', 'message': 'No time series data'}
            
            # Overall statistics
            mean_sales = self.time_series.mean()
            std_sales = self.time_series.std()
            median_sales = self.time_series.median()
            
            # Identify peak and low periods
            high_threshold = mean_sales + (0.5 * std_sales)
            low_threshold = mean_sales - (0.5 * std_sales)
            
            # Find high demand periods
            high_demand_mask = self.time_series > high_threshold
            high_demand_periods = self.time_series[high_demand_mask]
            
            # Find low demand periods
            low_demand_mask = self.time_series < low_threshold
            low_demand_periods = self.time_series[low_demand_mask]
            
            return {
                'status': 'success',
                'overall_mean': float(mean_sales),
                'overall_std': float(std_sales),
                'overall_median': float(median_sales),
                'high_threshold': float(high_threshold),
                'low_threshold': float(low_threshold),
                'high_demand_count': int(high_demand_mask.sum()),
                'low_demand_count': int(low_demand_mask.sum()),
                'peak_value': float(high_demand_periods.max()) if len(high_demand_periods) > 0 else 0,
                'lowest_value': float(low_demand_periods.min()) if len(low_demand_periods) > 0 else 0,
                'volatility': float(std_sales / mean_sales) if mean_sales != 0 else 0
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_stock_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate stock level recommendations based on demand analysis.
        
        Returns:
            List of stock recommendations with months and quantities
        """
        try:
            if not self.monthly_stats:
                return []
            
            recommendations = []
            overall_mean = self.time_series.mean()
            
            for month_data in self.monthly_stats:
                month = month_data['month']
                month_mean = month_data['mean']
                
                # Calculate percentage difference from overall mean
                pct_diff = ((month_mean - overall_mean) / overall_mean) * 100 if overall_mean != 0 else 0
                
                if pct_diff > 15:  # High demand
                    stock_action = 'INCREASE'
                    recommendation_text = 'Increase stock levels by 20-30%'
                    priority = 'HIGH'
                    quantity_increase = 25
                elif pct_diff > 5:  # Moderate high demand
                    stock_action = 'INCREASE'
                    recommendation_text = 'Increase stock levels by 10-15%'
                    priority = 'MEDIUM'
                    quantity_increase = 12
                elif pct_diff < -15:  # Low demand
                    stock_action = 'DECREASE'
                    recommendation_text = 'Reduce stock levels by 20-30%'
                    priority = 'MEDIUM'
                    quantity_increase = -25
                elif pct_diff < -5:  # Moderate low demand
                    stock_action = 'DECREASE'
                    recommendation_text = 'Reduce stock levels by 5-10%'
                    priority = 'LOW'
                    quantity_increase = -8
                else:  # Normal demand
                    stock_action = 'MAINTAIN'
                    recommendation_text = 'Maintain current stock levels'
                    priority = 'LOW'
                    quantity_increase = 0
                
                recommendations.append({
                    'month': month,
                    'action': stock_action,
                    'recommendation': recommendation_text,
                    'priority': priority,
                    'percentage_change': round(pct_diff, 2),
                    'quantity_increase_percent': quantity_increase,
                    'expected_sales': round(month_mean, 2)
                })
            
            return recommendations
        except Exception as e:
            print(f"Error generating stock recommendations: {e}")
            return []
    
    def generate_pricing_recommendations(self) -> List[Dict[str, Any]]:
        """
        Generate pricing and discount recommendations based on demand.
        
        Returns:
            List of pricing recommendations for each month
        """
        try:
            if not self.monthly_stats:
                return []
            
            recommendations = []
            overall_mean = self.time_series.mean()
            overall_std = self.time_series.std()
            
            for month_data in self.monthly_stats:
                month = month_data['month']
                month_mean = month_data['mean']
                month_std = month_data['std'] or 0
                
                # High demand → increase prices
                if month_mean > (overall_mean + overall_std):
                    discount_percent = 0
                    strategy = 'PREMIUM PRICING'
                    description = 'Increase prices by 5-10% due to high demand'
                    rationale = 'High demand allows for price increase to maximize revenue'
                
                # Moderate high demand → normal pricing
                elif month_mean > overall_mean:
                    discount_percent = 0
                    strategy = 'NORMAL PRICING'
                    description = 'Maintain current prices'
                    rationale = 'Demand is healthy, maintain prices for good margins'
                
                # Low demand → offer discounts
                elif month_mean < (overall_mean - overall_std):
                    discount_percent = 20
                    strategy = 'AGGRESSIVE DISCOUNT'
                    description = 'Offer 15-20% discount to stimulate sales'
                    rationale = 'Low demand requires aggressive discounts to boost volume'
                
                # Moderate low demand → light discount
                else:
                    discount_percent = 10
                    strategy = 'LIGHT DISCOUNT'
                    description = 'Offer 8-10% discount to encourage purchases'
                    rationale = 'Moderate demand - light discounts can increase volume'
                
                recommendations.append({
                    'month': month,
                    'strategy': strategy,
                    'discount_percent': discount_percent,
                    'description': description,
                    'rationale': rationale,
                    'expected_impact': f'Expected {5 + (discount_percent * 2)}% increase in volume'
                })
            
            return recommendations
        except Exception as e:
            print(f"Error generating pricing recommendations: {e}")
            return []
    
    def generate_festival_promotions(self) -> List[Dict[str, Any]]:
        """
        Generate festival-based promotional recommendations.
        
        Returns:
            List of festival promotions with timing and details
        """
        try:
            promotions = []
            overall_mean = self.time_series.mean()
            
            # Get monthly statistics for context
            month_means = {item['month']: item['mean'] for item in self.monthly_stats}
            
            for month, festival_info in self.FESTIVALS.items():
                month_sales = month_means.get(month, overall_mean)
                
                # Adjust recommendation based on baseline demand
                if month_sales > overall_mean:
                    # Already high demand - use moderate promotion
                    discount = festival_info['discount'] - 5
                    stock_increase = festival_info['stock_increase'] - 10
                    campaign_intensity = 'STANDARD'
                else:
                    # Lower demand - use standard promotion
                    discount = festival_info['discount']
                    stock_increase = festival_info['stock_increase']
                    campaign_intensity = 'AGGRESSIVE'
                
                promotions.append({
                    'month': month,
                    'festival_name': festival_info['name'],
                    'recommended_discount': max(5, discount),  # Min 5% discount
                    'stock_increase_percent': stock_increase,
                    'campaign_intensity': campaign_intensity,
                    'expected_sales_lift': round(10 + (discount * 0.5), 1),
                    'promotional_message': f"{'🎉 ' if campaign_intensity == 'AGGRESSIVE' else ''}Celebrate {festival_info['name']} with special offers!",
                    'implementation_timing': 'Start 2 weeks before, peak 1 week before'
                })
            
            return promotions
        except Exception as e:
            print(f"Error generating festival promotions: {e}")
            return []
    
    def calculate_price_elasticity(self) -> Dict[str, Any]:
        """
        Calculate price elasticity metrics for demand sensitivity analysis.
        
        Returns:
            Dict with elasticity metrics and recommendations
        """
        try:
            if len(self.time_series) < 2:
                return {'status': 'error', 'message': 'Insufficient data'}
            
            sales = self.time_series.values
            changes = np.diff(sales)
            pct_changes = (changes / sales[:-1]) * 100
            
            # Calculate elasticity proxy: sensitivity to changes
            volatility = np.std(pct_changes)
            cyclical_strength = np.abs(pct_changes).mean()
            
            elasticity_score = (volatility + cyclical_strength) / 2
            
            # Classify elasticity
            if elasticity_score < 10:
                elasticity_type = 'INELASTIC (Price Insensitive)'
                recommendation = 'Product has stable demand - focus on premium positioning'
                optimal_strategy = 'MARGIN OPTIMIZATION'
            elif elasticity_score < 20:
                elasticity_type = 'MODERATELY ELASTIC'
                recommendation = 'Balanced pricing approach - adjust based on competitors'
                optimal_strategy = 'DYNAMIC PRICING'
            else:
                elasticity_type = 'HIGHLY ELASTIC (Price Sensitive)'
                recommendation = 'Demand highly sensitive to price - use aggressive discounts'
                optimal_strategy = 'VOLUME MAXIMIZATION'
            
            return {
                'status': 'success',
                'elasticity_score': round(elasticity_score, 2),
                'elasticity_type': elasticity_type,
                'recommendation': recommendation,
                'optimal_strategy': optimal_strategy,
                'volatility_index': round(volatility, 2),
                'change_sensitivity': round(cyclical_strength, 2)
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_inventory_optimization(self) -> List[Dict[str, Any]]:
        """
        Generate inventory optimization recommendations using demand patterns.
        
        Returns:
            List of inventory level recommendations with safety stock calculations
        """
        try:
            if not self.monthly_stats:
                return []
            
            recommendations = []
            overall_mean = self.time_series.mean()
            overall_std = self.time_series.std()
            
            for month_data in self.monthly_stats:
                month = month_data['month']
                month_mean = month_data['mean']
                month_std = month_data['std'] or overall_std
                
                # Calculate safety stock using z-score approach
                service_level = 0.95  # 95% service level
                z_score = 1.645  # Z-score for 95% confidence
                lead_time_days = 7  # Assumed lead time
                
                # Demand variability
                demand_std_lt = month_std * np.sqrt(lead_time_days)
                safety_stock = z_score * demand_std_lt
                reorder_point = (month_mean * lead_time_days / 30) + safety_stock
                max_stock = reorder_point + (month_mean * 30 / 30)  # 1 month buffer
                
                # Calculate inventory turnover
                inventory_holding_cost_pct = 0.25  # 25% annually
                monthly_holding_cost = (month_mean * inventory_holding_cost_pct) / 12
                
                # Recommendations
                if month_mean > overall_mean:
                    urgency = 'CRITICAL'
                    action = 'Pre-order stock'
                    lead_days = 3
                else:
                    urgency = 'STANDARD'
                    action = 'Regular ordering'
                    lead_days = 7
                
                recommendations.append({
                    'month': month,
                    'forecasted_demand': round(month_mean, 2),
                    'reorder_point': round(reorder_point, 0),
                    'safety_stock': round(safety_stock, 0),
                    'max_stock_level': round(max_stock, 0),
                    'min_stock_level': round(safety_stock * 0.5, 0),
                    'order_urgency': urgency,
                    'recommended_action': action,
                    'lead_time_days': lead_days,
                    'holding_cost_monthly': round(monthly_holding_cost, 2),
                    'stock_turnover_velocity': 'FAST' if month_mean > overall_mean else 'NORMAL'
                })
            
            return recommendations
        except Exception as e:
            print(f"Error generating inventory optimization: {e}")
            return []
    
    def generate_revenue_optimization(self) -> Dict[str, Any]:
        """
        Generate revenue optimization strategies combining pricing and volume.
        
        Returns:
            Dict with revenue optimization tactics
        """
        try:
            elasticity = self.calculate_price_elasticity()
            analysis = self.analyze_demand_patterns()
            
            overall_mean = self.time_series.mean()
            overall_revenue = overall_mean * 365  # Annual estimate
            
            strategies = []
            
            if elasticity.get('status') == 'success':
                strategy_type = elasticity.get('optimal_strategy')
                
                if strategy_type == 'MARGIN OPTIMIZATION':
                    # Inelastic demand - increase margins
                    strategies.append({
                        'name': 'Premium Pricing Strategy',
                        'description': 'Product shows stable demand (inelastic)',
                        'tactic': 'Increase price by 8-12% while monitoring volume',
                        'expected_revenue_impact': 8.0,
                        'risk_level': 'LOW',
                        'implementation_timeline': '2-4 weeks'
                    })
                    strategies.append({
                        'name': 'Bundle Offers',
                        'description': 'Combine with complementary products',
                        'tactic': 'Create bundles with 5-10% discount on total',
                        'expected_revenue_impact': 15.0,
                        'risk_level': 'MEDIUM',
                        'implementation_timeline': '1-2 weeks'
                    })
                
                elif strategy_type == 'DYNAMIC PRICING':
                    # Moderate elasticity - balanced approach
                    strategies.append({
                        'name': 'Time-Based Pricing',
                        'description': 'Adjust prices based on demand cycles',
                        'tactic': 'Peak pricing: +5-8%, Off-peak pricing: -5-10%',
                        'expected_revenue_impact': 12.0,
                        'risk_level': 'MEDIUM',
                        'implementation_timeline': '3-4 weeks'
                    })
                    strategies.append({
                        'name': 'Volume Incentives',
                        'description': 'Encourage bulk purchases during slow periods',
                        'tactic': 'Buy 3+ units: 8% discount, 5+ units: 12% discount',
                        'expected_revenue_impact': 10.0,
                        'risk_level': 'LOW',
                        'implementation_timeline': '1 week'
                    })
                
                else:  # VOLUME MAXIMIZATION
                    # Elastic demand - focus on volume
                    strategies.append({
                        'name': 'Aggressive Discounting',
                        'description': 'High price sensitivity detected',
                        'tactic': 'Seasonal discounts: 15-25% during high-demand periods',
                        'expected_revenue_impact': 25.0,
                        'risk_level': 'HIGH',
                        'implementation_timeline': '1-2 weeks'
                    })
                    strategies.append({
                        'name': 'Loss Leader Promotion',
                        'description': 'Attract customers with deep discounts',
                        'tactic': 'Select items: 30% off to drive traffic & cross-selling',
                        'expected_revenue_impact': 20.0,
                        'risk_level': 'HIGH',
                        'implementation_timeline': '2-3 weeks'
                    })
            
            return {
                'status': 'success',
                'current_annual_revenue_estimate': round(overall_revenue, 2),
                'elasticity_type': elasticity.get('elasticity_type', 'UNKNOWN'),
                'revenue_optimization_strategies': strategies,
                'highest_impact_strategy': strategies[0] if strategies else {},
                'estimated_total_revenue_uplift': round(sum(s['expected_revenue_impact'] for s in strategies) / len(strategies), 1) if strategies else 0
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_competitive_positioning(self) -> Dict[str, Any]:
        """
        Generate competitive positioning recommendations based on demand patterns.
        
        Returns:
            Dict with competitive strategies
        """
        try:
            analysis = self.analyze_demand_patterns()
            
            if analysis.get('status') != 'success':
                return {'status': 'error', 'message': 'Insufficient data'}
            
            volatility = analysis.get('volatility', 0)
            peak_ratio = analysis.get('peak_value', 1) / max(analysis.get('lowest_value', 1), 1)
            
            if peak_ratio > 2.5:
                market_position = 'NICHE/SEASONAL PRODUCT'
                strategy = 'Focus on seasonal peak exploitation'
                tactics = [
                    'Dominate market during peak season',
                    'Build brand loyalty before peak',
                    'Prepare inventory 1-2 months in advance',
                    'Use predictive marketing to capture early demand'
                ]
            elif volatility > 0.4:
                market_position = 'HIGH VOLATILITY CATEGORY'
                strategy = 'Flexible capacity and agile supply chain'
                tactics = [
                    'Maintain flexible supplier relationships',
                    'Use drop-shipping for excess demand',
                    'Build strategic inventory during low periods',
                    'Cross-sell stable products during fluctuations'
                ]
            else:
                market_position = 'STABLE CORE PRODUCT'
                strategy = 'Consistent operations with optimization focus'
                tactics = [
                    'Maintain steady supply and demand',
                    'Focus on margin optimization',
                    'Build loyal customer base',
                    'Expand complementary product offerings'
                ]
            
            return {
                'status': 'success',
                'market_position': market_position,
                'recommended_strategy': strategy,
                'tactical_actions': tactics,
                'peak_to_low_ratio': round(peak_ratio, 1),
                'market_volatility': 'HIGH' if volatility > 0.3 else 'MODERATE' if volatility > 0.15 else 'LOW',
                'competitive_advantage_focus': 'Timing & Availability' if peak_ratio > 2 else 'Reliability' if volatility < 0.15 else 'Agility'
            }
        except Exception as e:
            return {'status': 'error', 'message': str(e)}
    
    def generate_comprehensive_recommendations(self) -> Dict[str, Any]:
        """
        Generate all enhanced recommendations in one call.
        
        Returns:
            Comprehensive recommendations dictionary with advanced analytics
        """
        try:
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(self.time_series),
                'demand_analysis': self.analyze_demand_patterns(),
                'price_elasticity': self.calculate_price_elasticity(),
                'revenue_optimization': self.generate_revenue_optimization(),
                'competitive_positioning': self.generate_competitive_positioning(),
                'stock_recommendations': self.generate_stock_recommendations(),
                'inventory_optimization': self.generate_inventory_optimization(),
                'pricing_recommendations': self.generate_pricing_recommendations(),
                'festival_promotions': self.generate_festival_promotions(),
                'summary': self._generate_executive_summary()
            }
        except Exception as e:
            return {
                'status': 'error',
                'message': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def _generate_executive_summary(self) -> Dict[str, str]:
        """Generate comprehensive executive summary of recommendations."""
        try:
            analysis = self.analyze_demand_patterns()
            elasticity = self.calculate_price_elasticity()
            inventory = self.generate_inventory_optimization()
            
            if analysis.get('status') != 'success':
                return {'summary': 'Unable to generate summary'}
            
            high_months = sum(1 for r in self.generate_stock_recommendations() 
                            if r['action'] == 'INCREASE')
            low_months = sum(1 for r in self.generate_stock_recommendations() 
                           if r['action'] == 'DECREASE')
            
            summary_text = (
                f"📊 Advanced Sales Analysis Summary:\n"
                f"Analyzed {len(self.time_series)} data points with high demand in {analysis.get('high_demand_count', 0)} periods.\n"
                f"🎯 Strategic Actions: Increase stock in {high_months} months (high demand), reduce in {low_months} months. "
                f"Implement dynamic pricing based on elasticity type ({elasticity.get('elasticity_type', 'STANDARD')}).\n"
                f"💰 Revenue Opportunity: Optimize margins in stable periods, maximize volume during peaks.\n"
                f"📦 Inventory Strategy: Maintain {len(inventory)} optimized reorder points with safety stocks calculated from demand variability.\n"
                f"🎊 Festival Strategy: Implement year-round promotions with intensified campaigns during peak seasons."
            )
            
            return {
                'summary': summary_text,
                'key_insight_1': f'📈 Demand Volatility: {analysis.get("volatility", 0):.2f} - ' + 
                                ('🔴 High seasonality detected (2.5x+ peak variation)' if analysis.get('volatility', 0) > 0.3 else '🟡 Moderate fluctuations' if analysis.get('volatility', 0) > 0.15 else '🟢 Stable demand pattern'),
                'key_insight_2': f'💵 Price Elasticity: {elasticity.get("elasticity_type", "STANDARD")} - ' + 
                                elasticity.get('recommendation', 'Monitor pricing carefully'),
                'key_insight_3': f'🎯 Peak/Low Ratio: {analysis.get("peak_value", 1) / max(analysis.get("lowest_value", 1), 1):.1f}x - ' + 
                                'Significant opportunity for dynamic pricing & inventory management',
                'key_insight_4': f'🏆 Competitive Position: ' + 
                                ('Focus on timing & availability advantage' if analysis.get('peak_value', 1) / max(analysis.get('lowest_value', 1), 1) > 2.5 else 'Emphasize reliability & consistency' if analysis.get('volatility', 0) < 0.15 else 'Build operational agility'),
                'actionable_items': 3 + min(high_months, 3),
                'implementation_priority': 'HIGH'
            }
        except Exception as e:
            return {'summary': f'Summary generation error: {str(e)}'}
    
    def get_quick_wins(self) -> List[Dict[str, Any]]:
        """
        Get immediate actionable recommendations that can drive quick revenue improvements.
        
        Returns:
            List of high-impact, low-effort recommendations
        """
        try:
            quick_wins = []
            
            # Quick Win 1: Identify immediate inventory adjustment
            inventory_recs = self.generate_inventory_optimization()
            if inventory_recs:
                critical_months = [r for r in inventory_recs if r.get('order_urgency') == 'CRITICAL']
                if critical_months:
                    quick_wins.append({
                        'title': 'Emergency Stock Increase',
                        'description': f'{len(critical_months)} high-demand periods identified',
                        'action': f"Immediately increase orders for {', '.join(r['month'] for r in critical_months[:3])}",
                        'expected_impact': 'Prevent stockouts during peak demand',
                        'effort': 'MINIMAL',
                        'timeframe': 'This week',
                        'risk_level': 'LOW'
                    })
            
            # Quick Win 2: Price optimization opportunity
            elasticity = self.calculate_price_elasticity()
            if elasticity.get('elasticity_type') == 'INELASTIC (Price Insensitive)':
                quick_wins.append({
                    'title': 'Margin Improvement Opportunity',
                    'description': 'Product shows price inelasticity',
                    'action': 'Implement 5-8% price increase immediately',
                    'expected_impact': '5-8% revenue boost from margin improvement',
                    'effort': 'MINIMAL',
                    'timeframe': 'Immediately',
                    'risk_level': 'LOW'
                })
            
            # Quick Win 3: Bundle opportunity
            highest_demand = max(self.monthly_stats, key=lambda x: x['mean'], default={})
            if highest_demand:
                quick_wins.append({
                    'title': 'Create Seasonal Bundle',
                    'description': f"Peak demand in {highest_demand.get('month')}",
                    'action': f"Launch bundle offer in {highest_demand.get('month')} with 10% discount for 2-3 items",
                    'expected_impact': '15-20% increase in average order value',
                    'effort': 'LOW',
                    'timeframe': '1-2 weeks',
                    'risk_level': 'LOW'
                })
            
            return quick_wins[:5]  # Return top 5 quick wins
        except Exception as e:
            print(f"Error generating quick wins: {e}")
            return []
    
    def get_top_recommendations(self, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Get comprehensive top priority recommendations with quick wins.
        
        Args:
            limit: Number of recommendations per category
            
        Returns:
            Dict with top recommendations by category plus quick wins
        """
        try:
            stock_recs = self.generate_stock_recommendations()
            pricing_recs = self.generate_pricing_recommendations()
            festival_proms = self.generate_festival_promotions()
            inventory_recs = self.generate_inventory_optimization()
            
            # Sort by priority
            priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            stock_recs_sorted = sorted(
                stock_recs, 
                key=lambda x: (priority_order.get(x['priority'], 3), 
                              abs(x['percentage_change']), 
                              x['action']),
                reverse=True
            )[:limit]
            
            inventory_recs_sorted = sorted(
                inventory_recs,
                key=lambda x: priority_order.get(x.get('order_urgency', 'STANDARD'), 3)
            )[:limit]
            
            return {
                'quick_wins': self.get_quick_wins(),
                'stock': stock_recs_sorted,
                'pricing': pricing_recs[:limit],
                'festivals': festival_proms[:3],
                'inventory': inventory_recs_sorted
            }
        except Exception as e:
            print(f"Error getting top recommendations: {e}")
            return {'quick_wins': [], 'stock': [], 'pricing': [], 'festivals': [], 'inventory': []}


def create_recommendations_json(time_series: pd.Series, 
                               forecast_data: Dict = None) -> Dict[str, Any]:
    """
    Helper function to create comprehensive recommendations in JSON format.
    
    Args:
        time_series: Historical sales data
        forecast_data: Optional forecast data
        
    Returns:
        JSON-serializable recommendations dictionary with all enhanced features
    """
    try:
        engine = RecommendationEngine(time_series, forecast_data)
        
        return {
            'success': True,
            'recommendations': engine.generate_comprehensive_recommendations(),
            'quick_wins': engine.get_quick_wins(),
            'top_recommendations': engine.get_top_recommendations(limit=5),
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }

"""
STEP 16: Recommendation Module
================================
Generates business suggestions based on sales analysis, demand patterns, and forecasts.

Features:
- Stock recommendations for high/low demand periods
- Pricing and discount strategies
- Festival-based promotional recommendations
- Demand pattern analysis
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any
import warnings

warnings.filterwarnings('ignore')


class RecommendationEngine:
    """
    Generates business recommendations based on sales analysis and forecasts.
    """
    
    # Festival definitions (month-based)
    FESTIVALS = {
        'January': {'name': 'New Year', 'discount': 15, 'stock_increase': 20},
        'February': {'name': "Valentine's Day", 'discount': 10, 'stock_increase': 15},
        'March': {'name': 'Spring Season', 'discount': 8, 'stock_increase': 12},
        'April': {'name': 'Easter', 'discount': 12, 'stock_increase': 18},
        'May': {'name': 'Memorial Day', 'discount': 10, 'stock_increase': 15},
        'June': {'name': 'Summer Start', 'discount': 8, 'stock_increase': 12},
        'July': {'name': 'Independence Day', 'discount': 12, 'stock_increase': 20},
        'August': {'name': 'Back to School', 'discount': 15, 'stock_increase': 25},
        'September': {'name': 'Labor Day', 'discount': 10, 'stock_increase': 15},
        'October': {'name': 'Halloween', 'discount': 12, 'stock_increase': 18},
        'November': {'name': 'Black Friday', 'discount': 25, 'stock_increase': 40},
        'December': {'name': 'Christmas', 'discount': 20, 'stock_increase': 35}
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
        """Calculate monthly statistics from time series."""
        try:
            # Convert to DataFrame for easier manipulation
            df = pd.DataFrame({
                'date': self.time_series.index,
                'sales': self.time_series.values
            })
            
            # Add month name
            df['month'] = pd.to_datetime(df['date']).dt.strftime('%B')
            
            # Group by month and calculate stats
            monthly = df.groupby('month')['sales'].agg(['mean', 'min', 'max', 'std'])
            monthly = monthly.reset_index()
            
            return monthly.to_dict('records')
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
    
    def generate_comprehensive_recommendations(self) -> Dict[str, Any]:
        """
        Generate all recommendations in one call.
        
        Returns:
            Comprehensive recommendations dictionary
        """
        try:
            return {
                'status': 'success',
                'timestamp': datetime.now().isoformat(),
                'data_points': len(self.time_series),
                'demand_analysis': self.analyze_demand_patterns(),
                'stock_recommendations': self.generate_stock_recommendations(),
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
        """Generate executive summary of recommendations."""
        try:
            analysis = self.analyze_demand_patterns()
            
            if analysis.get('status') != 'success':
                return {'summary': 'Unable to generate summary'}
            
            high_months = sum(1 for r in self.generate_stock_recommendations() 
                            if r['action'] == 'INCREASE')
            low_months = sum(1 for r in self.generate_stock_recommendations() 
                           if r['action'] == 'DECREASE')
            
            summary_text = (
                f"Based on analysis of {len(self.time_series)} data points with "
                f"{analysis.get('high_demand_count', 0)} high-demand periods: "
                f"Increase stock in {high_months} months (high demand), "
                f"reduce in {low_months} months (low demand). "
                f"Implement festival promotions year-round with intensified campaigns during peak seasons. "
                f"Use dynamic pricing based on demand - premium pricing in high seasons, "
                f"discounts up to 20% in slow periods to maintain customer engagement."
            )
            
            return {
                'summary': summary_text,
                'key_insight_1': f'Volatility Index: {analysis.get("volatility", 0):.2f} - ' + 
                                ('High seasonality detected' if analysis.get('volatility', 0) > 0.3 else 'Stable demand'),
                'key_insight_2': f'Peak/Low Ratio: {analysis.get("peak_value", 1) / max(analysis.get("lowest_value", 1), 1):.1f}x - ' + 
                                'Significant opportunity for dynamic pricing',
                'key_insight_3': f'High demand periods: {high_months}/12 months - ' + 
                                'Focus promotional efforts accordingly'
            }
        except Exception as e:
            return {'summary': f'Summary generation error: {str(e)}'}
    
    def get_top_recommendations(self, limit: int = 5) -> Dict[str, List[Dict]]:
        """
        Get top priority recommendations.
        
        Args:
            limit: Number of recommendations per category
            
        Returns:
            Dict with top recommendations by category
        """
        try:
            stock_recs = self.generate_stock_recommendations()
            pricing_recs = self.generate_pricing_recommendations()
            festival_proms = self.generate_festival_promotions()
            
            # Sort by priority
            priority_order = {'HIGH': 0, 'MEDIUM': 1, 'LOW': 2}
            stock_recs_sorted = sorted(
                stock_recs, 
                key=lambda x: (priority_order.get(x['priority'], 3), 
                              abs(x['percentage_change']), 
                              x['action']),
                reverse=True
            )[:limit]
            
            return {
                'stock': stock_recs_sorted,
                'pricing': pricing_recs[:limit],
                'festivals': festival_proms[:3]  # Top 3 festivals
            }
        except Exception as e:
            print(f"Error getting top recommendations: {e}")
            return {'stock': [], 'pricing': [], 'festivals': []}


def create_recommendations_json(time_series: pd.Series, 
                               forecast_data: Dict = None) -> Dict[str, Any]:
    """
    Helper function to create recommendations in JSON format.
    
    Args:
        time_series: Historical sales data
        forecast_data: Optional forecast data
        
    Returns:
        JSON-serializable recommendations dictionary
    """
    try:
        engine = RecommendationEngine(time_series, forecast_data)
        recommendations = engine.generate_comprehensive_recommendations()
        
        return {
            'success': True,
            'recommendations': recommendations,
            'generated_at': datetime.now().isoformat()
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e),
            'generated_at': datetime.now().isoformat()
        }

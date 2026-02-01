"""
STEP 16: Recommendation Engine Tests
====================================
Comprehensive test suite for the recommendation module.
"""

import unittest
import pandas as pd
import numpy as np
import json
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from recommendation_engine import RecommendationEngine, create_recommendations_json


class TestRecommendationEngine(unittest.TestCase):
    """Unit tests for RecommendationEngine class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test data for all tests."""
        # Load real Walmart dataset
        try:
            # Try multiple paths
            for path in [
                'Walmart-dataset.csv',
                'data/Walmart-dataset.csv',
                '../Walmart-dataset.csv',
                '/Users/sriyakaadhuluri/Documents/B.Tech/3rd_year/3-2/DEA/DEA-Project-1/Walmart-dataset.csv'
            ]:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            
            # Prepare data - aggregate to monthly
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            monthly_df = df.groupby(df['Date'].dt.to_period('M'))['Weekly_Sales'].sum()
            monthly_df.index = pd.to_datetime(monthly_df.index.to_timestamp())
            
            cls.time_series = monthly_df
            cls.has_data = True
            print(f"✅ Loaded {len(cls.time_series)} months of historical data")
            
        except Exception as e:
            print(f"⚠️ Could not load real data: {e}")
            # Create synthetic data
            dates = pd.date_range(start='2010-02-01', periods=36, freq='MS')
            # Create seasonal pattern
            base = 200000000
            seasonal = np.sin(np.arange(36) * 2 * np.pi / 12) * 50000000
            noise = np.random.normal(0, 20000000, 36)
            cls.time_series = pd.Series(base + seasonal + noise, index=dates)
            cls.has_data = False
            print(f"⚠️ Using synthetic data: {len(cls.time_series)} months")
    
    def test_1_engine_initialization(self):
        """Test RecommendationEngine initialization."""
        engine = RecommendationEngine(self.time_series)
        
        self.assertIsNotNone(engine)
        self.assertEqual(len(engine.time_series), len(self.time_series))
        self.assertIn('January', RecommendationEngine.FESTIVALS)
        self.assertIn('December', RecommendationEngine.FESTIVALS)
        
        print("✅ Engine initialization successful")
    
    def test_2_monthly_stats_calculation(self):
        """Test monthly statistics calculation."""
        engine = RecommendationEngine(self.time_series)
        stats = engine.monthly_stats
        
        self.assertIsInstance(stats, list)
        self.assertGreater(len(stats), 0)
        
        # Check structure
        if len(stats) > 0:
            self.assertIn('month', stats[0])
            self.assertIn('mean', stats[0])
            self.assertIn('min', stats[0])
            self.assertIn('max', stats[0])
        
        print(f"✅ Monthly stats calculated: {len(stats)} months analyzed")
    
    def test_3_demand_pattern_analysis(self):
        """Test demand pattern analysis."""
        engine = RecommendationEngine(self.time_series)
        analysis = engine.analyze_demand_patterns()
        
        self.assertEqual(analysis['status'], 'success')
        self.assertIn('overall_mean', analysis)
        self.assertIn('overall_std', analysis)
        self.assertIn('high_threshold', analysis)
        self.assertIn('low_threshold', analysis)
        self.assertGreaterEqual(analysis['high_threshold'], analysis['overall_mean'])
        self.assertLessEqual(analysis['low_threshold'], analysis['overall_mean'])
        
        print(f"✅ Demand analysis: Mean=${analysis['overall_mean']:,.0f}, " +
              f"Volatility={analysis['volatility']:.2f}")
    
    def test_4_stock_recommendations(self):
        """Test stock level recommendations generation."""
        engine = RecommendationEngine(self.time_series)
        recommendations = engine.generate_stock_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn('month', rec)
            self.assertIn('action', rec)
            self.assertIn('recommendation', rec)
            self.assertIn('priority', rec)
            self.assertIn('percentage_change', rec)
            
            # Validate action types
            self.assertIn(rec['action'], ['INCREASE', 'DECREASE', 'MAINTAIN'])
            self.assertIn(rec['priority'], ['HIGH', 'MEDIUM', 'LOW'])
        
        increase_count = sum(1 for r in recommendations if r['action'] == 'INCREASE')
        decrease_count = sum(1 for r in recommendations if r['action'] == 'DECREASE')
        
        print(f"✅ Stock recommendations: {increase_count} increase, " +
              f"{decrease_count} decrease, {12 - increase_count - decrease_count} maintain")
    
    def test_5_pricing_recommendations(self):
        """Test pricing and discount recommendations."""
        engine = RecommendationEngine(self.time_series)
        recommendations = engine.generate_pricing_recommendations()
        
        self.assertIsInstance(recommendations, list)
        self.assertGreater(len(recommendations), 0)
        
        for rec in recommendations:
            self.assertIn('month', rec)
            self.assertIn('strategy', rec)
            self.assertIn('discount_percent', rec)
            self.assertIn('description', rec)
            
            # Validate discount range
            self.assertGreaterEqual(rec['discount_percent'], 0)
            self.assertLessEqual(rec['discount_percent'], 25)
            
            # Validate strategy
            self.assertIn(rec['strategy'], 
                         ['PREMIUM PRICING', 'NORMAL PRICING', 'LIGHT DISCOUNT', 'AGGRESSIVE DISCOUNT'])
        
        avg_discount = np.mean([r['discount_percent'] for r in recommendations])
        print(f"✅ Pricing recommendations: Average discount={avg_discount:.1f}%")
    
    def test_6_festival_promotions(self):
        """Test festival-based promotional recommendations."""
        engine = RecommendationEngine(self.time_series)
        promotions = engine.generate_festival_promotions()
        
        self.assertEqual(len(promotions), 12)  # One per month
        
        for promo in promotions:
            self.assertIn('month', promo)
            self.assertIn('festival_name', promo)
            self.assertIn('recommended_discount', promo)
            self.assertIn('stock_increase_percent', promo)
            self.assertIn('campaign_intensity', promo)
            
            # Validate values
            self.assertGreater(promo['recommended_discount'], 0)
            self.assertGreater(promo['stock_increase_percent'], 0)
            self.assertIn(promo['campaign_intensity'], ['STANDARD', 'AGGRESSIVE'])
        
        # Check specific festivals
        black_friday = [p for p in promotions if p['festival_name'] == 'Black Friday'][0]
        christmas = [p for p in promotions if p['festival_name'] == 'Christmas'][0]
        
        self.assertGreaterEqual(black_friday['recommended_discount'], 15)
        self.assertGreaterEqual(christmas['recommended_discount'], 15)
        
        print(f"✅ Festival promotions: 12 months configured")
    
    def test_7_comprehensive_recommendations(self):
        """Test comprehensive recommendations generation."""
        engine = RecommendationEngine(self.time_series)
        recommendations = engine.generate_comprehensive_recommendations()
        
        self.assertEqual(recommendations['status'], 'success')
        self.assertIn('demand_analysis', recommendations)
        self.assertIn('stock_recommendations', recommendations)
        self.assertIn('pricing_recommendations', recommendations)
        self.assertIn('festival_promotions', recommendations)
        self.assertIn('summary', recommendations)
        
        # Validate each component
        self.assertGreater(len(recommendations['stock_recommendations']), 0)
        self.assertGreater(len(recommendations['pricing_recommendations']), 0)
        self.assertGreater(len(recommendations['festival_promotions']), 0)
        
        print("✅ Comprehensive recommendations generated")
    
    def test_8_executive_summary(self):
        """Test executive summary generation."""
        engine = RecommendationEngine(self.time_series)
        summary = engine._generate_executive_summary()
        
        self.assertIn('summary', summary)
        self.assertIn('key_insight_1', summary)
        self.assertIn('key_insight_2', summary)
        self.assertIn('key_insight_3', summary)
        
        self.assertGreater(len(summary['summary']), 50)
        
        print("✅ Executive summary generated successfully")
    
    def test_9_top_recommendations(self):
        """Test top priority recommendations extraction."""
        engine = RecommendationEngine(self.time_series)
        top_recs = engine.get_top_recommendations(limit=3)
        
        self.assertIn('stock', top_recs)
        self.assertIn('pricing', top_recs)
        self.assertIn('festivals', top_recs)
        
        self.assertLessEqual(len(top_recs['stock']), 3)
        self.assertLessEqual(len(top_recs['pricing']), 3)
        self.assertLessEqual(len(top_recs['festivals']), 3)
        
        print(f"✅ Top recommendations: {len(top_recs['stock'])} stock, " +
              f"{len(top_recs['pricing'])} pricing, {len(top_recs['festivals'])} festivals")
    
    def test_10_json_serialization(self):
        """Test JSON serialization of recommendations."""
        engine = RecommendationEngine(self.time_series)
        recommendations = engine.generate_comprehensive_recommendations()
        
        try:
            json_str = json.dumps(recommendations, default=str)
            self.assertGreater(len(json_str), 100)
            
            # Parse back to verify validity
            parsed = json.loads(json_str)
            self.assertEqual(parsed['status'], 'success')
            
            print(f"✅ JSON serialization successful ({len(json_str)} bytes)")
        except Exception as e:
            self.fail(f"JSON serialization failed: {e}")
    
    def test_11_error_handling_empty_data(self):
        """Test error handling with empty data."""
        empty_series = pd.Series([], dtype=float)
        engine = RecommendationEngine(empty_series)
        
        analysis = engine.analyze_demand_patterns()
        self.assertEqual(analysis['status'], 'error')
        
        print("✅ Error handling for empty data works correctly")
    
    def test_12_error_handling_invalid_data(self):
        """Test error handling with invalid data."""
        invalid_series = pd.Series([np.nan, np.inf, -np.inf, 0, 0, 0])
        engine = RecommendationEngine(invalid_series)
        
        # Should not crash
        recommendations = engine.generate_stock_recommendations()
        self.assertIsInstance(recommendations, list)
        
        print("✅ Error handling for invalid data works correctly")
    
    def test_13_helper_function(self):
        """Test create_recommendations_json helper function."""
        result = create_recommendations_json(self.time_series)
        
        self.assertTrue(result['success'])
        self.assertIn('recommendations', result)
        self.assertIn('generated_at', result)
        
        recs = result['recommendations']
        self.assertEqual(recs['status'], 'success')
        
        print("✅ Helper function works correctly")


class TestRecommendationIntegration(unittest.TestCase):
    """Integration tests for recommendation module."""
    
    @classmethod
    def setUpClass(cls):
        """Load data for integration tests."""
        try:
            for path in [
                'Walmart-dataset.csv',
                'data/Walmart-dataset.csv',
                '../Walmart-dataset.csv',
                '/Users/sriyakaadhuluri/Documents/B.Tech/3rd_year/3-2/DEA/DEA-Project-1/Walmart-dataset.csv'
            ]:
                if os.path.exists(path):
                    df = pd.read_csv(path)
                    break
            
            df['Date'] = pd.to_datetime(df['Date'], format='%d-%m-%Y')
            monthly_df = df.groupby(df['Date'].dt.to_period('M'))['Weekly_Sales'].sum()
            monthly_df.index = pd.to_datetime(monthly_df.index.to_timestamp())
            cls.time_series = monthly_df
            
        except:
            dates = pd.date_range(start='2010-02-01', periods=36, freq='MS')
            base = 200000000
            seasonal = np.sin(np.arange(36) * 2 * np.pi / 12) * 50000000
            noise = np.random.normal(0, 20000000, 36)
            cls.time_series = pd.Series(base + seasonal + noise, index=dates)
    
    def test_1_full_recommendation_pipeline(self):
        """Test complete recommendation generation pipeline."""
        engine = RecommendationEngine(self.time_series)
        
        # Step 1: Analyze demand
        analysis = engine.analyze_demand_patterns()
        self.assertEqual(analysis['status'], 'success')
        
        # Step 2: Generate stock recommendations
        stock_recs = engine.generate_stock_recommendations()
        self.assertGreater(len(stock_recs), 0)
        
        # Step 3: Generate pricing recommendations
        pricing_recs = engine.generate_pricing_recommendations()
        self.assertGreater(len(pricing_recs), 0)
        
        # Step 4: Generate festival promotions
        festivals = engine.generate_festival_promotions()
        self.assertEqual(len(festivals), 12)
        
        # Step 5: Generate comprehensive recommendations
        all_recs = engine.generate_comprehensive_recommendations()
        self.assertEqual(all_recs['status'], 'success')
        
        print("✅ Full pipeline executed successfully")
    
    def test_2_consistency_across_months(self):
        """Test consistency of recommendations across all months."""
        engine = RecommendationEngine(self.time_series)
        
        stock_recs = engine.generate_stock_recommendations()
        pricing_recs = engine.generate_pricing_recommendations()
        festivals = engine.generate_festival_promotions()
        
        # Check coverage
        stock_months = {r['month'] for r in stock_recs}
        pricing_months = {r['month'] for r in pricing_recs}
        festival_months = {r['month'] for r in festivals}
        
        expected_months = {'January', 'February', 'March', 'April', 'May', 'June',
                          'July', 'August', 'September', 'October', 'November', 'December'}
        
        self.assertEqual(stock_months, expected_months)
        self.assertEqual(pricing_months, expected_months)
        self.assertEqual(festival_months, expected_months)
        
        print("✅ Consistent coverage across all 12 months")
    
    def test_3_recommendation_alignment(self):
        """Test alignment between stock and pricing recommendations."""
        engine = RecommendationEngine(self.time_series)
        
        stock_recs = {r['month']: r for r in engine.generate_stock_recommendations()}
        pricing_recs = {r['month']: r for r in engine.generate_pricing_recommendations()}
        
        # In high demand months, stock should increase and pricing should be premium
        for month in stock_recs:
            stock_action = stock_recs[month]['action']
            pricing_strategy = pricing_recs[month]['strategy']
            
            if stock_action == 'INCREASE':
                # High demand could have premium or normal pricing
                self.assertIn(pricing_strategy, ['PREMIUM PRICING', 'NORMAL PRICING'])
            
            if stock_action == 'DECREASE':
                # Low demand should have discounts
                self.assertIn(pricing_strategy, ['LIGHT DISCOUNT', 'AGGRESSIVE DISCOUNT'])
        
        print("✅ Stock and pricing recommendations are aligned")


def run_tests():
    """Run all tests and display summary."""
    print("\n" + "="*70)
    print("🧪 STEP 16: RECOMMENDATION ENGINE TEST SUITE")
    print("="*70 + "\n")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add tests
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationEngine))
    suite.addTests(loader.loadTestsFromTestCase(TestRecommendationIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*70)
    print("📊 TEST SUMMARY")
    print("="*70)
    print(f"Tests Run: {result.testsRun}")
    print(f"Passed: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failed: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Pass Rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    print("="*70 + "\n")
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)

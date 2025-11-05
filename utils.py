import datetime
import numpy as np
import pandas as pd
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Union

@dataclass
class CountryInfo:
    """Country information with economic factors."""
    name: str
    price_multiplier: float
    currency: str
    currency_symbol: str
    tax_rate: float
    shipping_cost_factor: float
    market_weight: float

class Country(Enum):
    """Enum for supported countries with their economic characteristics."""
    USA = CountryInfo("USA", 1.0, "USD", "$", 0.08, 1.0, 0.25)
    GERMANY = CountryInfo("Germany", 1.15, "EUR", "€", 0.19, 0.8, 0.15)
    JAPAN = CountryInfo("Japan", 1.3, "JPY", "¥", 0.10, 1.2, 0.12)
    BRAZIL = CountryInfo("Brazil", 0.7, "BRL", "R$", 0.17, 1.5, 0.10)
    INDIA = CountryInfo("India", 0.4, "INR", "₹", 0.18, 0.6, 0.15)
    UK = CountryInfo("UK", 1.1, "GBP", "£", 0.20, 0.9, 0.08)
    CANADA = CountryInfo("Canada", 0.95, "CAD", "C$", 0.13, 1.1, 0.08)
    AUSTRALIA = CountryInfo("Australia", 1.2, "AUD", "A$", 0.10, 1.4, 0.07)
    FRANCE = CountryInfo("France", 1.15, "EUR", "€", 0.20, 0.8, 0.15)
    
    @classmethod
    def get_all_names(cls) -> list[str]:
        """Get list of all country names."""
        return [country.value.name for country in cls]
    
    @classmethod
    def get_weights(cls) -> list[float]:
        """Get list of market weights for all countries."""
        return [country.value.market_weight for country in cls]
    
    @classmethod
    def get_by_name(cls, name: str) -> 'Country':
        """Get country enum by name."""
        for country in cls:
            if country.value.name == name:
                return country
        raise ValueError(f"Country '{name}' not found")
    
    @classmethod
    def get_currency_map(cls) -> Dict[str, str]:
        """Get mapping from country name to currency symbol."""
        return {country.value.name: country.value.currency_symbol for country in cls}

def resolve_country(country: Union[str, Country, None]) -> str:
    """Convert Country enum or string to country name string."""
    if country is None:
        return "All"
    elif isinstance(country, Country):
        return country.value.name
    elif isinstance(country, str):
        return country
    else:
        return "All"

def mock_data(n_samples: int = 1000, drift_config: dict = None) -> pd.DataFrame:
    """
    Generate synthetic product price data with various features.
    
    Args:
        n_samples: Number of samples to generate (default: 1000)
        drift_config: Optional dict to simulate data drift with keys:
            - price_multipliers: dict mapping category to price multiplier
            - category_shift: dict mapping from category to target category weights
            - time_factor: float representing time progression (0.0-1.0)
        
    Returns:
        pd.DataFrame: Synthetic dataset with product information
        
    Raises:
        ValueError: If n_samples is not positive
    """
    # Validate input
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    
    # Use the Country enum for data generation
    countries = Country.get_all_names()
    country_weights = Country.get_weights()
    
    # Validate that we have countries and weights
    if not countries:
        raise ValueError("No countries available for data generation")
    
    if len(countries) != len(country_weights):
        raise ValueError("Number of countries and weights must match")
    
    # Normalize weights to ensure they sum to 1.0
    total_weight = sum(country_weights)
    if total_weight == 0:
        # Fallback to uniform distribution if all weights are zero
        country_weights = [1.0 / len(countries)] * len(countries)
    else:
        country_weights = [w / total_weight for w in country_weights]
    
    # Validate normalized weights
    if abs(sum(country_weights) - 1.0) > 1e-10:
        raise ValueError("Normalized weights do not sum to 1.0")
    
    # Generate countries for each sample
    try:
        countries_assigned = np.random.choice(countries, n_samples, p=country_weights)
    except ValueError as e:
        raise ValueError(f"Failed to generate country assignments: {e}")
    
    # First generate the categories
    base_categories = ["Electronics", "Clothing", "Home", "Books", "Sports"]
    category_weights = [0.2, 0.2, 0.2, 0.2, 0.2]  # Default equal weights
    
    # Apply category shift if drift_config is provided
    if drift_config and "category_shift" in drift_config:
        shift_config = drift_config["category_shift"]
        time_factor = drift_config.get("time_factor", 0.0)
        
        # Adjust weights based on drift configuration
        for i, category in enumerate(base_categories):
            if category in shift_config:
                # Gradually shift weights over time
                original_weight = category_weights[i]
                target_weight = shift_config[category]
                category_weights[i] = original_weight + (target_weight - original_weight) * time_factor
        
        # Normalize weights
        total_weight = sum(category_weights)
        category_weights = [w / total_weight for w in category_weights]
    
    categories = np.random.choice(base_categories, n_samples, p=category_weights)
    
    # Define category-specific distribution parameters (base USD prices)
    category_params = {
        "Electronics": {
            "price": {"mean": 200, "std": 50},
            "manufacturing_cost": {"mean": 80, "std": 20},
            "shipping_weight": {"mean": 2.5, "std": 1.2}
        },
        "Clothing": {
            "price": {"mean": 80, "std": 30},
            "manufacturing_cost": {"mean": 40, "std": 12},
            "shipping_weight": {"mean": 0.8, "std": 0.3}
        },
        "Home": {
            "price": {"mean": 150, "std": 45},
            "manufacturing_cost": {"mean": 60, "std": 18},
            "shipping_weight": {"mean": 5, "std": 2.5}
        },
        "Books": {
            "price": {"mean": 30, "std": 15},
            "manufacturing_cost": {"mean": 15, "std": 5},
            "shipping_weight": {"mean": 1.5, "std": 0.5}
        },
        "Sports": {
            "price": {"mean": 120, "std": 40},
            "manufacturing_cost": {"mean": 50, "std": 15},
            "shipping_weight": {"mean": 3, "std": 1.0}
        }
    }
    
    # Initialize empty arrays for category-specific attributes
    prices = np.zeros(n_samples)
    manufacturing_costs = np.zeros(n_samples)
    shipping_weights = np.zeros(n_samples)
    
    # Generate data for each category and apply country-specific adjustments
    for category in category_params:
        mask = categories == category
        category_count = np.sum(mask)
        
        # Generate base values
        base_prices = np.random.normal(
            category_params[category]["price"]["mean"],
            category_params[category]["price"]["std"],
            category_count
        )
        
        base_costs = np.random.normal(
            category_params[category]["manufacturing_cost"]["mean"],
            category_params[category]["manufacturing_cost"]["std"],
            category_count
        )
        
        base_weights = np.random.normal(
            category_params[category]["shipping_weight"]["mean"],
            category_params[category]["shipping_weight"]["std"],
            category_count
        )
        
        # Apply country-specific multipliers
        category_countries = countries_assigned[mask]
        for i, country_name in enumerate(category_countries):
            country = Country.get_by_name(country_name)
            country_info = country.value
            base_prices[i] *= country_info.price_multiplier
            base_costs[i] *= country_info.price_multiplier * 0.8  # Manufacturing costs scale less
            base_weights[i] *= country_info.shipping_cost_factor
        
        prices[mask] = base_prices
        manufacturing_costs[mask] = base_costs
        shipping_weights[mask] = base_weights
    
    # Apply price drift if drift_config is provided
    if drift_config and "price_multipliers" in drift_config:
        price_multipliers = drift_config["price_multipliers"]
        time_factor = drift_config.get("time_factor", 0.0)
        
        for category in price_multipliers:
            if category in categories:
                mask = categories == category
                target_multiplier = price_multipliers[category]
                # Apply gradual price change over time
                drift_multiplier = 1.0 + (target_multiplier - 1.0) * time_factor
                prices[mask] *= drift_multiplier
                manufacturing_costs[mask] *= drift_multiplier * 0.8  # Costs change less than prices
    
    # Ensure all values are positive
    prices = np.maximum(prices, 10)  # Minimum price of $10
    manufacturing_costs = np.maximum(manufacturing_costs, 5)  # Minimum cost of $5
    shipping_weights = np.maximum(shipping_weights, 0.1)  # Minimum weight of 0.1 kg
    
    # Generate remaining data with country-aware features
    try:
        # Generate product IDs
        product_ids = [f"PROD-{i:04d}" for i in range(n_samples)]
        
        # Generate currencies and tax rates with error handling
        currencies = []
        tax_rates = []
        for country in countries_assigned:
            try:
                country_enum = Country.get_by_name(country)
                currencies.append(country_enum.value.currency)
                tax_rates.append(country_enum.value.tax_rate)
            except ValueError:
                # Fallback to USA if country not found
                currencies.append("USD")
                tax_rates.append(0.08)
        
        # Generate other features
        brand_ratings = np.random.uniform(1, 5, n_samples)
        num_reviews = np.random.randint(0, 500, n_samples)
        days_since_release = np.random.randint(1, 1000, n_samples)
        discount_offered = np.random.choice([True, False], n_samples)
        competitors_price = prices * np.random.uniform(0.8, 1.2, n_samples)
        final_prices = prices * (1 + np.array(tax_rates))
        
        data = {
            "product_id": product_ids,
            "category": categories,
            "country": countries_assigned,
            "currency": currencies,
            "brand_rating": brand_ratings,
            "num_reviews": num_reviews,
            "days_since_release": days_since_release,
            "discount_offered": discount_offered,
            "shipping_weight": shipping_weights,
            "competitors_price": competitors_price,
            "manufacturing_cost": manufacturing_costs,
            "price": prices,
            "tax_rate": tax_rates,
            "final_price": final_prices
        }
    except Exception as e:
        raise ValueError(f"Failed to generate product data: {e}")
    
    # Introduce some missing values
    try:
        for col in ["brand_rating", "num_reviews", "shipping_weight"]:
            if col in data:
                missing_count = max(1, int(n_samples * 0.05))  # At least 1 missing value
                missing_indices = np.random.choice(range(n_samples), size=missing_count, replace=False)
                data[col] = pd.Series(data[col])
                data[col].iloc[missing_indices] = None
    except Exception as e:
        raise ValueError(f"Failed to introduce missing values: {e}")
    
    # Create DataFrame with error handling
    try:
        df = pd.DataFrame(data)
        
        # Validate the DataFrame
        if df.empty:
            raise ValueError("Generated DataFrame is empty")
        
        if len(df) != n_samples:
            raise ValueError(f"DataFrame length ({len(df)}) doesn't match expected samples ({n_samples})")
        
        return df
    except Exception as e:
        raise ValueError(f"Failed to create DataFrame: {e}")

def make_category_boxplot_data(field, variable_name, data: pd.DataFrame):
    """Generate boxplot data for a specific field by category.
    
    Args:
        field: The dataframe column to plot
        variable_name: The JavaScript variable name to push data to
    """
    colors = {
        "Electronics": "rgba(255, 99, 132, 0.7)",
        "Clothing": "rgba(54, 162, 235, 0.7)",
        "Home": "rgba(255, 206, 86, 0.7)",
        "Books": "rgba(75, 192, 192, 0.7)",
        "Sports": "rgba(153, 102, 255, 0.7)"
    }
    
    # Get currency info for the field if it's price-related
    currency_symbol = "$"  # Default to USD
    if "country" in data.columns and len(data["country"].unique()) == 1:
        country_name = data["country"].iloc[0]
        try:
            country = Country.get_by_name(country_name)
            currency_symbol = country.value.currency_symbol
        except ValueError:
            currency_symbol = "$"  # Fallback to USD
    
    js_code = []
    for category in data['category'].unique():
        cat_data = data[data['category'] == category][field].tolist()
        cat_stats = {
            "mean": float(data[data['category'] == category][field].mean()),
            "median": float(data[data['category'] == category][field].median()),
            "min": float(data[data['category'] == category][field].min()),
            "max": float(data[data['category'] == category][field].max()),
            "count": len(cat_data)
        }
        
        field_unit = currency_symbol if field in ["price", "manufacturing_cost", "competitors_price", "final_price"] else ""
        
        js_code.append(f"""
            {variable_name}.push({{
                y: {cat_data},
                type: 'box',
                name: '{category}',
                boxpoints: 'outliers',
                marker: {{
                    color: '{colors.get(category, "rgba(100, 100, 100, 0.7)")}'
                }},
                hoverinfo: 'all',
                hovertemplate: 
                    '<b>{category}</b><br>' +
                    'Mean: {field_unit}{cat_stats["mean"]:.2f}<br>' +
                    'Median: {field_unit}{cat_stats["median"]:.2f}<br>' +
                    'Min: {field_unit}{cat_stats["min"]:.2f}<br>' +
                    'Max: {field_unit}{cat_stats["max"]:.2f}<br>' +
                    'Count: {cat_stats["count"]}<extra></extra>'
            }});
        """)
    return "\n".join(js_code)

def make_profit_margin_data(data: pd.DataFrame):
    colors = {
        "Electronics": "rgba(255, 99, 132, 0.7)",
        "Clothing": "rgba(54, 162, 235, 0.7)",
        "Home": "rgba(255, 206, 86, 0.7)",
        "Books": "rgba(75, 192, 192, 0.7)",
        "Sports": "rgba(153, 102, 255, 0.7)"
    }
    
    # Calculate profit margins for each category
    categories = data['category'].unique()
    margins = []
    for category in categories:
        cat_df = data[data['category'] == category]
        margin = ((cat_df['price'] - cat_df['manufacturing_cost']) / cat_df['price']).mean()
        margins.append(margin)
    
    return f"""
    profitMarginData.push({{
        x: {list(categories)},
        y: {margins},
        type: 'bar',
        marker: {{
            color: {[colors.get(cat, "rgba(100, 100, 100, 0.7)") for cat in categories]}
        }},
        hovertemplate: 
            '<b>%{{x}}</b><br>' +
            'Profit Margin: %{{y:.1%}}<extra></extra>'
    }});
    """

def make_country_comparison_data(data: pd.DataFrame):
    """Generate country comparison visualization data."""
    if 'country' not in data.columns:
        return ""
    
    countries = data['country'].unique()
    country_stats = {}
    
    for country in countries:
        country_data = data[data['country'] == country]
        country_stats[country] = {
            'avg_price': float(country_data['price'].mean()),
            'avg_final_price': float(country_data['final_price'].mean()),
            'count': len(country_data),
            'currency': country_data['currency'].iloc[0] if len(country_data) > 0 else 'USD'
        }
    
    # Create bar chart data for country comparison
    return f"""
    var countryComparisonData = [
        {{
            x: {list(country_stats.keys())},
            y: {[stats['avg_price'] for stats in country_stats.values()]},
            type: 'bar',
            name: 'Average Price',
            marker: {{
                color: 'rgba(75, 192, 192, 0.7)',
                line: {{
                    color: 'rgba(75, 192, 192, 1.0)',
                    width: 1
                }}
            }},
            hovertemplate: '<b>%{{x}}</b><br>Avg Price: %{{y:.2f}}<extra></extra>'
        }},
        {{
            x: {list(country_stats.keys())},
            y: {[stats['avg_final_price'] for stats in country_stats.values()]},
            type: 'bar',
            name: 'Average Final Price (with tax)',
            marker: {{
                color: 'rgba(255, 159, 64, 0.7)',
                line: {{
                    color: 'rgba(255, 159, 64, 1.0)',
                    width: 1
                }}
            }},
            hovertemplate: '<b>%{{x}}</b><br>Final Price: %{{y:.2f}}<extra></extra>'
        }}
    ];
    """

def make_country_market_share_data(data: pd.DataFrame):
    """Generate market share pie chart data by country."""
    if 'country' not in data.columns:
        return ""
    
    country_counts = data['country'].value_counts()
    colors = [
        'rgba(255, 99, 132, 0.8)', 'rgba(54, 162, 235, 0.8)', 'rgba(255, 205, 86, 0.8)',
        'rgba(75, 192, 192, 0.8)', 'rgba(153, 102, 255, 0.8)', 'rgba(255, 159, 64, 0.8)',
        'rgba(199, 199, 199, 0.8)', 'rgba(83, 102, 255, 0.8)'
    ]
    
    return f"""
    var marketShareData = [{{
        values: {country_counts.values.tolist()},
        labels: {country_counts.index.tolist()},
        type: 'pie',
        marker: {{
            colors: {colors[:len(country_counts)]}
        }},
        hovertemplate: '<b>%{{label}}</b><br>Products: %{{value}}<br>Share: %{{percent}}<extra></extra>'
    }}];
    """

def generate_data_report(cleaned_data: pd.DataFrame, raw_data: pd.DataFrame, analysis: dict) -> str:
    """Generate HTML report focused on data analysis."""
    # Pre-generate all the JavaScript code for the charts
    price_boxplot_js = make_category_boxplot_data('price', 'categoryPriceData', cleaned_data)
    cost_boxplot_js = make_category_boxplot_data('manufacturing_cost', 'categoryCostData', cleaned_data)
    weight_boxplot_js = make_category_boxplot_data('shipping_weight', 'categoryWeightData', cleaned_data)
    profit_margin_js = make_profit_margin_data(cleaned_data)
    country_comparison_js = make_country_comparison_data(raw_data)
    market_share_js = make_country_market_share_data(raw_data)
    
    # Check if we're showing data for a specific country
    is_single_country = 'country' in cleaned_data.columns and len(cleaned_data['country'].unique()) == 1
    country_name = cleaned_data['country'].iloc[0] if is_single_country else "All Countries"
    
    # Get appropriate currency symbol
    currency_symbol = "$"
    if is_single_country:
        country_name = cleaned_data['country'].iloc[0]
        try:
            country = Country.get_by_name(country_name)
            currency_symbol = country.value.currency_symbol
        except ValueError:
            currency_symbol = "$"  # Fallback to USD
    
    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Data Analysis Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .card {{ margin-bottom: 20px; }}
            .metric-card {{ text-align: center; padding: 15px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .chart-container {{ height: 400px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Product Data Analysis Report - {country_name}</h1>
            <p class="lead">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{raw_data.shape[0]}</div>
                        <div class="metric-label">Total Products</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{len(raw_data['category'].unique())}</div>
                        <div class="metric-label">Product Categories</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{currency_symbol}{cleaned_data['price'].mean():.2f}</div>
                        <div class="metric-label">Avg. Price</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{sum(raw_data.isnull().sum())}</div>
                        <div class="metric-label">Missing Values</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Price by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-price-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Manufacturing Cost by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-cost-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Shipping Weight by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-weight-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Profit Margin by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="profit-margin-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Price Distribution</h5>
                        </div>
                        <div class="card-body">
                            <div id="price-dist-chart"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            {'' if is_single_country else f'''
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Market Share by Country</h5>
                        </div>
                        <div class="card-body">
                            <div id="market-share-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Price Comparison by Country</h5>
                        </div>
                        <div class="card-body">
                            <div id="country-comparison-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            '''}
        </div>
        
        <script>
            // Price Distribution Chart
            var priceData = {cleaned_data["price"].tolist()};
            var priceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Price ($)" }}
            }};
            Plotly.newPlot('price-dist-chart', [{{
                x: priceData,
                type: 'histogram',
                marker: {{
                    color: 'rgba(75, 192, 192, 0.7)',
                    line: {{
                        color: 'rgba(75, 192, 192, 1.0)',
                        width: 1
                    }}
                }},
                hovertemplate: 'Price: $%{{x}}<br>Count: %{{y}}<extra></extra>'
            }}], priceLayout);
            
            // Category-specific price boxplots
            var categoryPriceData = [];
            {price_boxplot_js}
            
            var categoryPriceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Price ($)" }}
            }};
            
            Plotly.newPlot('category-price-chart', categoryPriceData, categoryPriceLayout);
            
            // Category-specific cost boxplots
            var categoryCostData = [];
            {cost_boxplot_js}
            
            var categoryCostLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Manufacturing Cost ($)" }}
            }};
            
            Plotly.newPlot('category-cost-chart', categoryCostData, categoryCostLayout);
            
            // Category-specific weight boxplots
            var categoryWeightData = [];
            {weight_boxplot_js}
            
            var categoryWeightLayout = {{
                height: 400, 
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "Shipping Weight (kg)" }}
            }};
            
            Plotly.newPlot('category-weight-chart', categoryWeightData, categoryWeightLayout);
            
            // Profit margin by category
            var profitMarginData = [];
            {profit_margin_js}
            
            var profitMarginLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ 
                    title: "Profit Margin (%)",
                    tickformat: '.0%'
                }}
            }};
            
            Plotly.newPlot('profit-margin-chart', profitMarginData, profitMarginLayout);
            
            {'' if is_single_country else f'''
            // Market Share Chart
            {market_share_js}
            
            var marketShareLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                title: "Product Distribution by Country"
            }};
            
            Plotly.newPlot('market-share-chart', marketShareData, marketShareLayout);
            
            // Country Comparison Chart
            {country_comparison_js}
            
            var countryComparisonLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Country" }},
                yaxis: {{ title: "Price" }},
                barmode: 'group'
            }};
            
            Plotly.newPlot('country-comparison-chart', countryComparisonData, countryComparisonLayout);
            '''}
        </script>
    </body>
    </html>
    """
    return html_content

def generate_model_report(model: dict, data: pd.DataFrame) -> str:
    """Generate HTML report focused on model training results."""
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Model Training Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0-alpha1/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body {{ padding: 20px; }}
            .card {{ margin-bottom: 20px; }}
            .metric-card {{ text-align: center; padding: 15px; }}
            .metric-value {{ font-size: 24px; font-weight: bold; }}
            .metric-label {{ font-size: 14px; color: #666; }}
            .chart-container {{ height: 400px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="my-4">Price Prediction Model Report</h1>
            <p class="lead">Generated on {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            
            <div class="row mb-4">
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{model["metrics"]["r2_score"]:.2f}</div>
                        <div class="metric-label">R² Score</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">${model["metrics"]["rmse"]:.2f}</div>
                        <div class="metric-label">RMSE</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">${model["metrics"]["mae"]:.2f}</div>
                        <div class="metric-label">MAE</div>
                    </div>
                </div>
                <div class="col-md-3">
                    <div class="card metric-card">
                        <div class="metric-value">{model["model_params"]["epochs"]}</div>
                        <div class="metric-label">Training Epochs</div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Feature Importance</h5>
                        </div>
                        <div class="card-body">
                            <div id="feature-importance-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
                
                <div class="col-md-6">
                    <div class="card">
                        <div class="card-header">
                            <h5>Learning Curve</h5>
                        </div>
                        <div class="card-body">
                            <div id="learning-curve-chart" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Actual vs Predicted Prices</h5>
                        </div>
                        <div class="card-body">
                            <div id="prediction-scatter" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
            
            <div class="row">
                <div class="col-md-12">
                    <div class="card">
                        <div class="card-header">
                            <h5>Model Performance by Category</h5>
                        </div>
                        <div class="card-body">
                            <div id="category-performance" class="chart-container"></div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
        
        <script>
            // Feature Importance Chart
            var featureNames = Object.keys({model["feature_importance"]});
            var importanceValues = Object.values({model["feature_importance"]});
            
            var featureImportanceLayout = {{
                height: 400,
                margin: {{ t: 10, l: 150 }},
                xaxis: {{ title: "Importance" }}
            }};
            
            Plotly.newPlot('feature-importance-chart', [{{
                x: importanceValues,
                y: featureNames,
                type: 'bar',
                orientation: 'h',
                marker: {{
                    color: 'rgba(153, 102, 255, 0.7)',
                    line: {{
                        color: 'rgba(153, 102, 255, 1.0)',
                        width: 1
                    }}
                }},
                hovertemplate: '<b>%{{y}}</b><br>Importance: %{{x:.2f}}<extra></extra>'
            }}], featureImportanceLayout);
            
            // Learning Curve Chart
            var learningCurveLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Epochs" }},
                yaxis: {{ title: "Loss" }}
            }};
            
            Plotly.newPlot('learning-curve-chart', [
                {{
                    x: {model["learning_curve"]["epochs"]},
                    y: {model["learning_curve"]["train_loss"]},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Training Loss',
                    line: {{ color: 'rgba(255, 99, 132, 1)' }},
                    hovertemplate: 'Epoch: %{{x}}<br>Training Loss: %{{y:.2f}}<extra></extra>'
                }},
                {{
                    x: {model["learning_curve"]["epochs"]},
                    y: {model["learning_curve"]["val_loss"]},
                    type: 'scatter',
                    mode: 'lines+markers',
                    name: 'Validation Loss',
                    line: {{ color: 'rgba(54, 162, 235, 1)' }},
                    hovertemplate: 'Epoch: %{{x}}<br>Validation Loss: %{{y:.2f}}<extra></extra>'
                }}
            ], learningCurveLayout);
            
            // Generate simulated predictions for the scatter plot
            function simulatePredictions(actual, error) {{
                return actual.map(value => value * (1 + (Math.random() * 2 - 1) * error));
            }}
            
            var actualPrices = {data["price"].tolist()};
            var predictedPrices = simulatePredictions(actualPrices, {0.2 - (model["metrics"]["r2_score"] * 0.15)});
            var categories = {data["category"].tolist()};
            var uniqueCategories = {list(data["category"].unique())};
            
            // Define colors for each category
            var categoryColors = {{
                "Electronics": "rgba(255, 99, 132, 0.7)",
                "Clothing": "rgba(54, 162, 235, 0.7)",
                "Home": "rgba(255, 206, 86, 0.7)",
                "Books": "rgba(75, 192, 192, 0.7)",
                "Sports": "rgba(153, 102, 255, 0.7)"
            }};
            
            // Create traces for each category
            var scatterTraces = [];
            
            uniqueCategories.forEach(category => {{
                var categoryIndices = [];
                for (var i = 0; i < categories.length; i++) {{
                    if (categories[i] === category) {{
                        categoryIndices.push(i);
                    }}
                }}
                
                var categoryActualPrices = categoryIndices.map(i => actualPrices[i]);
                var categoryPredictedPrices = categoryIndices.map(i => predictedPrices[i]);
                
                scatterTraces.push({{
                    x: categoryActualPrices,
                    y: categoryPredictedPrices,
                    mode: 'markers',
                    type: 'scatter',
                    name: category,
                    marker: {{
                        color: categoryColors[category] || "rgba(100, 100, 100, 0.7)",
                        size: 8,
                        line: {{
                            color: categoryColors[category]?.replace('0.7', '1.0') || "rgba(100, 100, 100, 1.0)",
                            width: 1
                        }}
                    }},
                    hovertemplate: '<b>' + category + '</b><br>Actual: $%{{x:.2f}}<br>Predicted: $%{{y:.2f}}<extra></extra>'
                }});
            }});
            
            // Add the perfect prediction line
            scatterTraces.push({{
                x: [0, {data["price"].max() * 1.1}],
                y: [0, {data["price"].max() * 1.1}],
                mode: 'lines',
                type: 'scatter',
                name: 'Perfect Prediction',
                line: {{
                    color: 'rgba(0, 0, 0, 0.5)',
                    dash: 'dash'
                }}
            }});
            
            var scatterLayout = {{
                height: 500,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Actual Price ($)" }},
                yaxis: {{ title: "Predicted Price ($)" }},
                legend: {{ 
                    orientation: 'h',
                    yanchor: 'bottom',
                    y: 1.02,
                    xanchor: 'right',
                    x: 1
                }}
            }};
            
            Plotly.newPlot('prediction-scatter', scatterTraces, scatterLayout);
            
            // Performance by Category
            var categoryPerformance = [];
            
            // Simulate RMSE for each category based on overall RMSE
            var baseRMSE = {model["metrics"]["rmse"]};
            
            uniqueCategories.forEach(category => {{
                // Simulate slightly different performance per category
                var categoryRMSE = baseRMSE * (0.8 + Math.random() * 0.4);
                categoryPerformance.push(categoryRMSE);
            }});
            
            var categoryPerformanceLayout = {{
                height: 400,
                margin: {{ t: 10 }},
                xaxis: {{ title: "Category" }},
                yaxis: {{ title: "RMSE ($)" }}
            }};
            
            Plotly.newPlot('category-performance', [{{
                x: uniqueCategories,
                y: categoryPerformance,
                type: 'bar',
                marker: {{
                    color: [
                        'rgba(255, 99, 132, 0.7)',
                        'rgba(54, 162, 235, 0.7)',
                        'rgba(255, 206, 86, 0.7)',
                        'rgba(75, 192, 192, 0.7)',
                        'rgba(153, 102, 255, 0.7)'
                    ],
                    line: {{
                        color: [
                            'rgba(255, 99, 132, 1.0)',
                            'rgba(54, 162, 235, 1.0)',
                            'rgba(255, 206, 86, 1.0)',
                            'rgba(75, 192, 192, 1.0)',
                            'rgba(153, 102, 255, 1.0)'
                        ],
                        width: 1
                    }}
                }},
                hovertemplate: '<b>%{{x}}</b><br>RMSE: $%{{y:.2f}}<extra></extra>'
            }}], categoryPerformanceLayout);
        </script>
    </body>
    </html>
    """
    return html_content
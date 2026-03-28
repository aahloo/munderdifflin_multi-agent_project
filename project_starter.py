import pandas as pd
import numpy as np
import os
import time
import dotenv
import ast
from sqlalchemy.sql import text
from datetime import datetime, timedelta
from typing import Dict, List, Union
from sqlalchemy import create_engine, Engine

# Create an SQLite database
db_engine = create_engine("sqlite:///munder_difflin.db")

# List containing the different kinds of papers 
paper_supplies = [
    # Paper Types (priced per sheet unless specified)
    {"item_name": "A4 paper",                         "category": "paper",        "unit_price": 0.05},
    {"item_name": "Letter-sized paper",              "category": "paper",        "unit_price": 0.06},
    {"item_name": "Cardstock",                        "category": "paper",        "unit_price": 0.15},
    {"item_name": "Colored paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Glossy paper",                     "category": "paper",        "unit_price": 0.20},
    {"item_name": "Matte paper",                      "category": "paper",        "unit_price": 0.18},
    {"item_name": "Recycled paper",                   "category": "paper",        "unit_price": 0.08},
    {"item_name": "Eco-friendly paper",               "category": "paper",        "unit_price": 0.12},
    {"item_name": "Poster paper",                     "category": "paper",        "unit_price": 0.25},
    {"item_name": "Banner paper",                     "category": "paper",        "unit_price": 0.30},
    {"item_name": "Kraft paper",                      "category": "paper",        "unit_price": 0.10},
    {"item_name": "Construction paper",               "category": "paper",        "unit_price": 0.07},
    {"item_name": "Wrapping paper",                   "category": "paper",        "unit_price": 0.15},
    {"item_name": "Glitter paper",                    "category": "paper",        "unit_price": 0.22},
    {"item_name": "Decorative paper",                 "category": "paper",        "unit_price": 0.18},
    {"item_name": "Letterhead paper",                 "category": "paper",        "unit_price": 0.12},
    {"item_name": "Legal-size paper",                 "category": "paper",        "unit_price": 0.08},
    {"item_name": "Crepe paper",                      "category": "paper",        "unit_price": 0.05},
    {"item_name": "Photo paper",                      "category": "paper",        "unit_price": 0.25},
    {"item_name": "Uncoated paper",                   "category": "paper",        "unit_price": 0.06},
    {"item_name": "Butcher paper",                    "category": "paper",        "unit_price": 0.10},
    {"item_name": "Heavyweight paper",                "category": "paper",        "unit_price": 0.20},
    {"item_name": "Standard copy paper",              "category": "paper",        "unit_price": 0.04},
    {"item_name": "Bright-colored paper",             "category": "paper",        "unit_price": 0.12},
    {"item_name": "Patterned paper",                  "category": "paper",        "unit_price": 0.15},

    # Product Types (priced per unit)
    {"item_name": "Paper plates",                     "category": "product",      "unit_price": 0.10},  # per plate
    {"item_name": "Paper cups",                       "category": "product",      "unit_price": 0.08},  # per cup
    {"item_name": "Paper napkins",                    "category": "product",      "unit_price": 0.02},  # per napkin
    {"item_name": "Disposable cups",                  "category": "product",      "unit_price": 0.10},  # per cup
    {"item_name": "Table covers",                     "category": "product",      "unit_price": 1.50},  # per cover
    {"item_name": "Envelopes",                        "category": "product",      "unit_price": 0.05},  # per envelope
    {"item_name": "Sticky notes",                     "category": "product",      "unit_price": 0.03},  # per sheet
    {"item_name": "Notepads",                         "category": "product",      "unit_price": 2.00},  # per pad
    {"item_name": "Invitation cards",                 "category": "product",      "unit_price": 0.50},  # per card
    {"item_name": "Flyers",                           "category": "product",      "unit_price": 0.15},  # per flyer
    {"item_name": "Party streamers",                  "category": "product",      "unit_price": 0.05},  # per roll
    {"item_name": "Decorative adhesive tape (washi tape)", "category": "product", "unit_price": 0.20},  # per roll
    {"item_name": "Paper party bags",                 "category": "product",      "unit_price": 0.25},  # per bag
    {"item_name": "Name tags with lanyards",          "category": "product",      "unit_price": 0.75},  # per tag
    {"item_name": "Presentation folders",             "category": "product",      "unit_price": 0.50},  # per folder

    # Large-format items (priced per unit)
    {"item_name": "Large poster paper (24x36 inches)", "category": "large_format", "unit_price": 1.00},
    {"item_name": "Rolls of banner paper (36-inch width)", "category": "large_format", "unit_price": 2.50},

    # Specialty papers
    {"item_name": "100 lb cover stock",               "category": "specialty",    "unit_price": 0.50},
    {"item_name": "80 lb text paper",                 "category": "specialty",    "unit_price": 0.40},
    {"item_name": "250 gsm cardstock",                "category": "specialty",    "unit_price": 0.30},
    {"item_name": "220 gsm poster paper",             "category": "specialty",    "unit_price": 0.35},
]

# Given below are some utility functions you can use to implement your multi-agent system

def generate_sample_inventory(paper_supplies: list, coverage: float = 0.4, seed: int = 137) -> pd.DataFrame:
    """
    Generate inventory for exactly a specified percentage of items from the full paper supply list.

    This function randomly selects exactly `coverage` × N items from the `paper_supplies` list,
    and assigns each selected item:
    - a random stock quantity between 200 and 800,
    - a minimum stock level between 50 and 150.

    The random seed ensures reproducibility of selection and stock levels.

    Args:
        paper_supplies (list): A list of dictionaries, each representing a paper item with
                               keys 'item_name', 'category', and 'unit_price'.
        coverage (float, optional): Fraction of items to include in the inventory (default is 0.4, or 40%).
        seed (int, optional): Random seed for reproducibility (default is 137).

    Returns:
        pd.DataFrame: A DataFrame with the selected items and assigned inventory values, including:
                      - item_name
                      - category
                      - unit_price
                      - current_stock
                      - min_stock_level
    """
    # Ensure reproducible random output
    np.random.seed(seed)

    # Calculate number of items to include based on coverage
    num_items = int(len(paper_supplies) * coverage)

    # Randomly select item indices without replacement
    selected_indices = np.random.choice(
        range(len(paper_supplies)),
        size=num_items,
        replace=False
    )

    # Extract selected items from paper_supplies list
    selected_items = [paper_supplies[i] for i in selected_indices]

    # Construct inventory records
    inventory = []
    for item in selected_items:
        inventory.append({
            "item_name": item["item_name"],
            "category": item["category"],
            "unit_price": item["unit_price"],
            "current_stock": np.random.randint(200, 800),  # Realistic stock range
            "min_stock_level": np.random.randint(50, 150)  # Reasonable threshold for reordering
        })

    # Return inventory as a pandas DataFrame
    return pd.DataFrame(inventory)

def init_database(db_engine: Engine, seed: int = 137) -> Engine:    
    """
    Set up the Munder Difflin database with all required tables and initial records.

    This function performs the following tasks:
    - Creates the 'transactions' table for logging stock orders and sales
    - Loads customer inquiries from 'quote_requests.csv' into a 'quote_requests' table
    - Loads previous quotes from 'quotes.csv' into a 'quotes' table, extracting useful metadata
    - Generates a random subset of paper inventory using `generate_sample_inventory`
    - Inserts initial financial records including available cash and starting stock levels

    Args:
        db_engine (Engine): A SQLAlchemy engine connected to the SQLite database.
        seed (int, optional): A random seed used to control reproducibility of inventory stock levels.
                              Default is 137.

    Returns:
        Engine: The same SQLAlchemy engine, after initializing all necessary tables and records.

    Raises:
        Exception: If an error occurs during setup, the exception is printed and raised.
    """
    try:
        # ----------------------------
        # 1. Create an empty 'transactions' table schema
        # ----------------------------
        transactions_schema = pd.DataFrame({
            "id": [],
            "item_name": [],
            "transaction_type": [],  # 'stock_orders' or 'sales'
            "units": [],             # Quantity involved
            "price": [],             # Total price for the transaction
            "transaction_date": [],  # ISO-formatted date
        })
        transactions_schema.to_sql("transactions", db_engine, if_exists="replace", index=False)

        # Set a consistent starting date
        initial_date = datetime(2025, 1, 1).isoformat()

        # ----------------------------
        # 2. Load and initialize 'quote_requests' table
        # ----------------------------
        quote_requests_df = pd.read_csv("quote_requests.csv")
        quote_requests_df["id"] = range(1, len(quote_requests_df) + 1)
        quote_requests_df.to_sql("quote_requests", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 3. Load and transform 'quotes' table
        # ----------------------------
        quotes_df = pd.read_csv("quotes.csv")
        quotes_df["request_id"] = range(1, len(quotes_df) + 1)
        quotes_df["order_date"] = initial_date

        # Unpack metadata fields (job_type, order_size, event_type) if present
        if "request_metadata" in quotes_df.columns:
            quotes_df["request_metadata"] = quotes_df["request_metadata"].apply(
                lambda x: ast.literal_eval(x) if isinstance(x, str) else x
            )
            quotes_df["job_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("job_type", ""))
            quotes_df["order_size"] = quotes_df["request_metadata"].apply(lambda x: x.get("order_size", ""))
            quotes_df["event_type"] = quotes_df["request_metadata"].apply(lambda x: x.get("event_type", ""))

        # Retain only relevant columns
        quotes_df = quotes_df[[
            "request_id",
            "total_amount",
            "quote_explanation",
            "order_date",
            "job_type",
            "order_size",
            "event_type"
        ]]
        quotes_df.to_sql("quotes", db_engine, if_exists="replace", index=False)

        # ----------------------------
        # 4. Generate inventory and seed stock
        # ----------------------------
        inventory_df = generate_sample_inventory(paper_supplies, seed=seed)

        # Seed initial transactions
        initial_transactions = []

        # Add a starting cash balance via a dummy sales transaction
        initial_transactions.append({
            "item_name": None,
            "transaction_type": "sales",
            "units": None,
            "price": 50000.0,
            "transaction_date": initial_date,
        })

        # Add one stock order transaction per inventory item
        for _, item in inventory_df.iterrows():
            initial_transactions.append({
                "item_name": item["item_name"],
                "transaction_type": "stock_orders",
                "units": item["current_stock"],
                "price": item["current_stock"] * item["unit_price"],
                "transaction_date": initial_date,
            })

        # Commit transactions to database
        pd.DataFrame(initial_transactions).to_sql("transactions", db_engine, if_exists="append", index=False)

        # Save the inventory reference table
        inventory_df.to_sql("inventory", db_engine, if_exists="replace", index=False)

        return db_engine

    except Exception as e:
        print(f"Error initializing database: {e}")
        raise

def create_transaction(
    item_name: str,
    transaction_type: str,
    quantity: int,
    price: float,
    date: Union[str, datetime],
) -> int:
    """
    This function records a transaction of type 'stock_orders' or 'sales' with a specified
    item name, quantity, total price, and transaction date into the 'transactions' table of the database.

    Args:
        item_name (str): The name of the item involved in the transaction.
        transaction_type (str): Either 'stock_orders' or 'sales'.
        quantity (int): Number of units involved in the transaction.
        price (float): Total price of the transaction.
        date (str or datetime): Date of the transaction in ISO 8601 format.

    Returns:
        int: The ID of the newly inserted transaction.

    Raises:
        ValueError: If `transaction_type` is not 'stock_orders' or 'sales'.
        Exception: For other database or execution errors.
    """
    try:
        # Convert datetime to ISO string if necessary
        date_str = date.isoformat() if isinstance(date, datetime) else date

        # Validate transaction type
        if transaction_type not in {"stock_orders", "sales"}:
            raise ValueError("Transaction type must be 'stock_orders' or 'sales'")

        # Prepare transaction record as a single-row DataFrame
        transaction = pd.DataFrame([{
            "item_name": item_name,
            "transaction_type": transaction_type,
            "units": quantity,
            "price": price,
            "transaction_date": date_str,
        }])

        # Insert the record into the database
        transaction.to_sql("transactions", db_engine, if_exists="append", index=False)

        # Fetch and return the ID of the inserted row
        result = pd.read_sql("SELECT last_insert_rowid() as id", db_engine)
        return int(result.iloc[0]["id"])

    except Exception as e:
        print(f"Error creating transaction: {e}")
        raise

def get_all_inventory(as_of_date: str) -> Dict[str, int]:
    """
    Retrieve a snapshot of available inventory as of a specific date.

    This function calculates the net quantity of each item by summing 
    all stock orders and subtracting all sales up to and including the given date.

    Only items with positive stock are included in the result.

    Args:
        as_of_date (str): ISO-formatted date string (YYYY-MM-DD) representing the inventory cutoff.

    Returns:
        Dict[str, int]: A dictionary mapping item names to their current stock levels.
    """
    # SQL query to compute stock levels per item as of the given date
    query = """
        SELECT
            item_name,
            SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END) as stock
        FROM transactions
        WHERE item_name IS NOT NULL
        AND transaction_date <= :as_of_date
        GROUP BY item_name
        HAVING stock > 0
    """

    # Execute the query with the date parameter
    result = pd.read_sql(query, db_engine, params={"as_of_date": as_of_date})

    # Convert the result into a dictionary {item_name: stock}
    return dict(zip(result["item_name"], result["stock"]))

def get_stock_level(item_name: str, as_of_date: Union[str, datetime]) -> pd.DataFrame:
    """
    Retrieve the stock level of a specific item as of a given date.

    This function calculates the net stock by summing all 'stock_orders' and 
    subtracting all 'sales' transactions for the specified item up to the given date.

    Args:
        item_name (str): The name of the item to look up.
        as_of_date (str or datetime): The cutoff date (inclusive) for calculating stock.

    Returns:
        pd.DataFrame: A single-row DataFrame with columns 'item_name' and 'current_stock'.
    """
    # Convert date to ISO string format if it's a datetime object
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # SQL query to compute net stock level for the item
    stock_query = """
        SELECT
            item_name,
            COALESCE(SUM(CASE
                WHEN transaction_type = 'stock_orders' THEN units
                WHEN transaction_type = 'sales' THEN -units
                ELSE 0
            END), 0) AS current_stock
        FROM transactions
        WHERE item_name = :item_name
        AND transaction_date <= :as_of_date
    """

    # Execute query and return result as a DataFrame
    return pd.read_sql(
        stock_query,
        db_engine,
        params={"item_name": item_name, "as_of_date": as_of_date},
    )

def get_supplier_delivery_date(input_date_str: str, quantity: int) -> str:
    """
    Estimate the supplier delivery date based on the requested order quantity and a starting date.

    Delivery lead time increases with order size:
        - ≤10 units: same day
        - 11–100 units: 1 day
        - 101–1000 units: 4 days
        - >1000 units: 7 days

    Args:
        input_date_str (str): The starting date in ISO format (YYYY-MM-DD).
        quantity (int): The number of units in the order.

    Returns:
        str: Estimated delivery date in ISO format (YYYY-MM-DD).
    """
    # Debug log (comment out in production if needed)
    print(f"FUNC (get_supplier_delivery_date): Calculating for qty {quantity} from date string '{input_date_str}'")

    # Attempt to parse the input date
    try:
        input_date_dt = datetime.fromisoformat(input_date_str.split("T")[0])
    except (ValueError, TypeError):
        # Fallback to current date on format error
        print(f"WARN (get_supplier_delivery_date): Invalid date format '{input_date_str}', using today as base.")
        input_date_dt = datetime.now()

    # Determine delivery delay based on quantity
    if quantity <= 10:
        days = 0
    elif quantity <= 100:
        days = 1
    elif quantity <= 1000:
        days = 4
    else:
        days = 7

    # Add delivery days to the starting date
    delivery_date_dt = input_date_dt + timedelta(days=days)

    # Return formatted delivery date
    return delivery_date_dt.strftime("%Y-%m-%d")

def get_cash_balance(as_of_date: Union[str, datetime]) -> float:
    """
    Calculate the current cash balance as of a specified date.

    The balance is computed by subtracting total stock purchase costs ('stock_orders')
    from total revenue ('sales') recorded in the transactions table up to the given date.

    Args:
        as_of_date (str or datetime): The cutoff date (inclusive) in ISO format or as a datetime object.

    Returns:
        float: Net cash balance as of the given date. Returns 0.0 if no transactions exist or an error occurs.
    """
    try:
        # Convert date to ISO format if it's a datetime object
        if isinstance(as_of_date, datetime):
            as_of_date = as_of_date.isoformat()

        # Query all transactions on or before the specified date
        transactions = pd.read_sql(
            "SELECT * FROM transactions WHERE transaction_date <= :as_of_date",
            db_engine,
            params={"as_of_date": as_of_date},
        )

        # Compute the difference between sales and stock purchases
        if not transactions.empty:
            total_sales = transactions.loc[transactions["transaction_type"] == "sales", "price"].sum()
            total_purchases = transactions.loc[transactions["transaction_type"] == "stock_orders", "price"].sum()
            return float(total_sales - total_purchases)

        return 0.0

    except Exception as e:
        print(f"Error getting cash balance: {e}")
        return 0.0


def generate_financial_report(as_of_date: Union[str, datetime]) -> Dict:
    """
    Generate a complete financial report for the company as of a specific date.

    This includes:
    - Cash balance
    - Inventory valuation
    - Combined asset total
    - Itemized inventory breakdown
    - Top 5 best-selling products

    Args:
        as_of_date (str or datetime): The date (inclusive) for which to generate the report.

    Returns:
        Dict: A dictionary containing the financial report fields:
            - 'as_of_date': The date of the report
            - 'cash_balance': Total cash available
            - 'inventory_value': Total value of inventory
            - 'total_assets': Combined cash and inventory value
            - 'inventory_summary': List of items with stock and valuation details
            - 'top_selling_products': List of top 5 products by revenue
    """
    # Normalize date input
    if isinstance(as_of_date, datetime):
        as_of_date = as_of_date.isoformat()

    # Get current cash balance
    cash = get_cash_balance(as_of_date)

    # Get current inventory snapshot
    inventory_df = pd.read_sql("SELECT * FROM inventory", db_engine)
    inventory_value = 0.0
    inventory_summary = []

    # Compute total inventory value and summary by item
    for _, item in inventory_df.iterrows():
        stock_info = get_stock_level(item["item_name"], as_of_date)
        stock = stock_info["current_stock"].iloc[0]
        item_value = stock * item["unit_price"]
        inventory_value += item_value

        inventory_summary.append({
            "item_name": item["item_name"],
            "stock": stock,
            "unit_price": item["unit_price"],
            "value": item_value,
        })

    # Identify top-selling products by revenue
    top_sales_query = """
        SELECT item_name, SUM(units) as total_units, SUM(price) as total_revenue
        FROM transactions
        WHERE transaction_type = 'sales' AND transaction_date <= :date
        GROUP BY item_name
        ORDER BY total_revenue DESC
        LIMIT 5
    """
    top_sales = pd.read_sql(top_sales_query, db_engine, params={"date": as_of_date})
    top_selling_products = top_sales.to_dict(orient="records")

    return {
        "as_of_date": as_of_date,
        "cash_balance": cash,
        "inventory_value": inventory_value,
        "total_assets": cash + inventory_value,
        "inventory_summary": inventory_summary,
        "top_selling_products": top_selling_products,
    }


def search_quote_history(search_terms: List[str], limit: int = 5) -> List[Dict]:
    """
    Retrieve a list of historical quotes that match any of the provided search terms.

    The function searches both the original customer request (from `quote_requests`) and
    the explanation for the quote (from `quotes`) for each keyword. Results are sorted by
    most recent order date and limited by the `limit` parameter.

    Args:
        search_terms (List[str]): List of terms to match against customer requests and explanations.
        limit (int, optional): Maximum number of quote records to return. Default is 5.

    Returns:
        List[Dict]: A list of matching quotes, each represented as a dictionary with fields:
            - original_request
            - total_amount
            - quote_explanation
            - job_type
            - order_size
            - event_type
            - order_date
    """
    conditions = []
    params = {}

    # Build SQL WHERE clause using LIKE filters for each search term
    for i, term in enumerate(search_terms):
        param_name = f"term_{i}"
        conditions.append(
            f"(LOWER(qr.response) LIKE :{param_name} OR "
            f"LOWER(q.quote_explanation) LIKE :{param_name})"
        )
        params[param_name] = f"%{term.lower()}%"

    # Combine conditions; fallback to always-true if no terms provided
    where_clause = " AND ".join(conditions) if conditions else "1=1"

    # Final SQL query to join quotes with quote_requests
    query = f"""
        SELECT
            qr.response AS original_request,
            q.total_amount,
            q.quote_explanation,
            q.job_type,
            q.order_size,
            q.event_type,
            q.order_date
        FROM quotes q
        JOIN quote_requests qr ON q.request_id = qr.id
        WHERE {where_clause}
        ORDER BY q.order_date DESC
        LIMIT {limit}
    """

    # Execute parameterized query
    with db_engine.connect() as conn:
        result = conn.execute(text(query), params)
        return [dict(row._mapping) for row in result]

########################
########################
########################
# YOUR MULTI AGENT STARTS HERE
########################
########################
########################

# using smolagents framework for this multi-agent system workflow
from smolagents import ToolCallingAgent, OpenAIServerModel, tool

# Set up and load your env parameters and instantiate your model.
dotenv.load_dotenv()
openai_api_key = os.getenv('UDACITY_OPENAI_API_KEY')

model = OpenAIServerModel(
    model_id='gpt-4o-mini',
    api_base='https://openai.vocareum.com/v1',
    api_key=openai_api_key,
)

"""Set up tools for your agents to use, these should be methods that combine the database functions above
 and apply criteria to them to ensure that the flow of the system is correct."""


# Tools for inventory agent (3 implemented)

# one to check all inventory, one to check stock level of a specific item, 
# and one to evaluate reorder needs based on current inventory levels
@tool 
def check_all_inventory(as_of_date: str) -> str:

    """
    Get a complete snapshot of all available inventory as of a given date.
    
    Args:
        as_of_date: Date string in YYYY-MM-DD format.
    
    Returns:
        Formatted string listing all items with stock quantities above zero.
    """

    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return "No inventory data available."
    
    lines = [f"  {item}: {qty} units" for item, qty in sorted(inventory.items())]
    return "Current Inventory:\n" + "\n".join(lines)

@tool
def check_stock_level(item_name: str, as_of_date: str) -> str:
    
    """
    Get the current stock level and catalog unit price for one specific item.
    
    Args:
        item_name: Exact name of the inventory item to check.
        as_of_date: Date string in YYYY-MM-DD format.

    Returns:
        String reporting stock quantity and catalog unit pricefor the requested item.
    """

    result = get_stock_level(item_name, as_of_date)
    qty = int(result["current_stock"].iloc[0])

    # Look up catalog price for the item
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": item_name}
    )

    if not price_df.empty:
        unit_price = float(price_df["unit_price"].iloc[0])
        price_str = f"Catalog price: ${unit_price:.4f} per unit."
    else:
        matches = [p for p in paper_supplies if p["item_name"] == item_name]
        unit_price = matches[0]["unit_price"] if matches else None
        price_str = f"Catalog price: ${unit_price:.4f} per unit." if unit_price else "Item not in catalog. Default price: $0.10 / unit."

    if qty == 0:
        return f"'{item_name}' is currently out of stock as of {as_of_date}. {price_str}"
    
    return f"'{item_name}': {qty} units available as of {as_of_date}. {price_str}"

@tool
def evaluate_reorder_needs(as_of_date: str) -> str:

    """
    Evaulate current inventory against minimum stock levels and identify items needing reorder.
    
    Args:
        as_of_date: Date string in YYYY-MM-DD format.
        
    Returns:
        Report of items below minimum stock threshold and list of well-stocked alternatives.
    """

    inventory = get_all_inventory(as_of_date)
    if not inventory:
        return "No inventory data available."
    
    low_stock = []
    adequate_stock = []

    for item, qty in inventory.items():
        if qty == 0:
            low_stock.append(f"  OUT OF STOCK: {item}")
        elif qty < 100:
            low_stock.append(f"  LOW ({qty} units): {item}")
        else:
            adequate_stock.append(f"{item} ({qty} units)")
    
    lines = []

    if low_stock:
        lines.append("Items needing attention:\n" + "\n".join(low_stock))
    else:
        lines.append(f"All tracked items are above minimum stock levels.")

    if adequate_stock:
        lines.append(f"Well-stocked items available for immediate fulfillment: {', '.join(adequate_stock)}")

    return "\n".join(lines)


# Tools for quoting agent

# Two tools employed: one to search past quotes for similar orders, and one to get a 
# financial summary to inform pricing decisions
@tool
def search_past_quotes(keywords: str) -> str:

    """
    Search historical quote records to find pricing precedents for similar orders.
    
    Args:
        keywords: Space-separated keywords describing the order type or product.
    Returns:
        Formatted string of matching historical quotes with amounts and explanations.
    """

    terms = [k.strip() for k in keywords.split() if len(k.strip()) > 2][:2]  # Limit to 2 meaningful keywords
    if not terms:
        terms = [keywords.strip()] # Fallback to using the whole input if no valid keywords extracted

    results = search_quote_history(terms, limit=5)

    if not results:
        return f"No historical quotes found matching '{keywords}'."
    
    lines = [
        f"  ${q['total_amount']:.2f} - {q['quote_explanation'][:120]}"
        for q in results if q.get('total_amount', -1) > 0
    ]

    if not lines:
        return f"No valid historical quotes found for '{keywords}'."
    
    return f"Historical pricing for '{keywords}':\n" + "\n".join(lines)

@tool
def get_financial_summary(as_of_date: str) -> str:

    """
    Get a financial summary including cash balance, inventory value, and top sellers.
    
    Args:
        as_of_date: Date string in YYYY-MM-DD format.
        
    Returns:
        Formatted financial summary to inform pricing and discount decisions.
    """

    report = generate_financial_report(as_of_date)
    cash = get_cash_balance(as_of_date)
    top_sellers = [p["item_name"] for p in report.get("top_selling_products", [])[:3] if p.get("item_name")]
    top_str = ", ".join(top_sellers) if top_sellers else "No sales data"

    return (
        f"Financial Summary ({as_of_date}):\n"
        f"  Cash Balance    : ${cash:.2f}\n"
        f"  Inventory Value : ${report.get('inventory_value', 0):,.2f}\n"
        f"  Total Assets    : ${report.get('total_assets', 0):,.2f}\n"
        f"  Top Sellers     : {top_str}"
    )

@tool
def get_catalog_price(item_name: str) -> str:
    
    """
    Look up the retail unit price for an item from the catelog.
    
    Args:
        item_name: Exact name of the item to look up.
    
    Returns:
        The retail unit price per unit for the specified item.
    """

    # Check inventory table first
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": item_name}
    )

    if not price_df.empty:
        unit_price = float(price_df["unit_price"].iloc[0])
        return f"'{item_name}' catalog price: ${unit_price:.4f} per unit."
    
    # Fall back to paper_supplies catalog if not found in inventory
    matches = [p for p in paper_supplies if p["item_name"] == item_name]
    if matches:
        unit_price = matches[0]["unit_price"]
        return f"'{item_name}' catalog price: ${unit_price:.4f} per unit."
    
    # will terminate search with a usable price if the item is not found in either the inventory 
    # or the original catalog list to prevent max steps from triggering and terminating loop
    return f"'{item_name}' not found in catalog. Use default price: $0.10 per unit for quoting purposes."

# Tools for ordering agent

# Two tools implemented: one to verify stock availability before confirming a sale, 
# and one to record a completed sale transaction
@tool
def verify_stock_for_sale(item_name: str, quantity: int, as_of_date: str) -> str:

    """
    Verify that sufficient stock exists before recording a sale.
    
    Args:
        item_name: Exact name of the item to sell.
        quantity: Number of units the customer is ordering.
        as_of_date: Date string in YYYY-MM-DD format.
    Returns:
        Confirmation of stock availability or explanation of shortfall.
    """

    result = get_stock_level(item_name, as_of_date)
    available_stock = int(result["current_stock"].iloc[0])

    if available_stock >= quantity:
        return f"Stock confirmed: {available_stock} units of '{item_name}' available; {quantity} can be sold."
    
    return (
        f"Insufficient stock: only {available_stock} units of '{item_name}' are available, "
        f"but {quantity} were requested."
    )

@tool
def record_sale(item_name: str, quantity: int, total_price: float, transaction_date: str) -> str:

    """
    Record a completed sale transaction, reducing inventory and adding revenue.
    
    Args:
        item_name: Exact name of the item being sold.
        quantity: Number of units sold.
        total_price: Total price for the entire order after any discounts have been applied.
        transaction_date: Transaction date in YYYY-MM-DD format.

    Returns:
        Sale confirmation with unit price and total revenue amount.
    """

    if total_price <= 0:
        return (
            f"Sale rejected: total_price must be greater than 0. "
            f"Received ${total_price:.2f}. "
            f"Retrieve the catalog price with get_catalog_price and recalculate before recording this sale."
        )
    
    existing = pd.read_sql(
        "SELECT COUNT(*) as cnt FROM transactions WHERE item_name = :name AND date(transaction_date) = date(:date)" \
        " AND transaction_type = 'sales' AND units = :qty",
        db_engine,
        params={"name": item_name, "date": transaction_date, "qty": quantity}
    )

    if int(existing["cnt"].iloc[0]) > 0:
        unit_price = total_price / quantity if quantity > 0 else 0
        return (
            f"ALREADY RECORDED: Sale of {quantity} units of '{item_name}' on {transaction_date} "
            f"was already completed (${unit_price:.4f} / unit = ${total_price:.2f}). "
            f"Call final_answer now - do not call record_sale again."
        )

    result = get_stock_level(item_name, transaction_date)
    available_stock = int(result["current_stock"].iloc[0])

    if available_stock < quantity:
        return (
            f"Sale rejected: only {available_stock} units of '{item_name}' are available, "
            f"but requested {quantity} units. No transaction recorded."
        )    

    create_transaction(
        item_name=item_name,
        transaction_type="sales",
        quantity=quantity,
        price=total_price,
        date=transaction_date
    )

    unit_price = total_price / quantity if quantity > 0 else 0

    return (
        f"SALE COMPLETE: {quantity} units of '{item_name}' @ ${unit_price:.4f} / unit = ${total_price:.2f}."
        f"Inventory reduced by {quantity} units. Call final_answer now with this confirmation."
    )


# Tools for procurement agent

# Three tools implemented: one to estimate delivery times for supplier orders, one to assess restock urgency,
# and one to place a supplier order and record the transaction

@tool
def get_delivery_estimate(quantity: int, order_date: str) -> str:

    """
    Estimate the supplier delivery date for a given quantity ordered on a given date.
    
    Args:
        quantity: Number of units to order from the supplier.
        order_date: Date the order is placed in YYYY-MM-DD format.
        
    Returns:
        Estimated delivery date from the supplier.
    """

    delivery_date = get_supplier_delivery_date(order_date, quantity)

    return (
        f"Estimated delivery date for {quantity} units: {delivery_date}. "
        f"(<= 10: same day | 11 - 100 +1 day | 101 - 1,000: +4 days | > 1000: +7 days)"
    )

@tool
def assess_restock_urgency(item_name: str, as_of_date: str) -> str:

    """
    Assess whether a specific item needs urgent restocking based on current stock level.
    
    Args:
        item_name: Exact name of the item to evaluate.
        as_of_date: Date string in YYYY-MM-DD format for current stock evaluation.
        
    Returns:
        Urgency assessment with current stock level.
    """

    result = get_stock_level(item_name, as_of_date)
    qty = int(result["current_stock"].iloc[0])

    if qty == 0:
        return f"URGENT: '{item_name}' is completely out of stock. Immediate reorder needed."
    elif qty < 100:
        return f"LOW: '{item_name}' has only {qty} units left - reorder soon."
    
    return f"ADEQUATE: '{item_name}' has {qty} units in stock - no immediate restock needed."

@tool
def place_supplier_order(item_name: str, quantity: int, transaction_date: str) -> str:

    """
    Place a stock replenishment order with the supplier and record the purchase.
    Supplier cost is calculated automatically at 60% of the item's retail unit price.
    
    Args:
        item_name: Exact name of the item to restock.
        quantity: Numebr of units to order from the supplier.        
        transaction_date: Date of the purchase order in YYYY-MM-DD format.
        
    Returns:
        Purchase confirmation with total cost and expected delivery date.
    """

    # Retrieve the unit price from the inventory reference table
    price_df = pd.read_sql(
        "SELECT unit_price FROM inventory WHERE item_name = :name",
        db_engine,
        params={"name": item_name}
    )

    if price_df.empty: # Fall back to paper_supplies catalog
        matches = [p for p in paper_supplies if p["item_name"] == item_name]
        unit_price = matches[0]["unit_price"] if matches else 0.10
    else:
        unit_price = float(price_df["unit_price"].iloc[0])

    cost_per_unit = unit_price * 0.60
    total_cost = quantity * cost_per_unit

    create_transaction(
        item_name=item_name,
        transaction_type="stock_orders",
        quantity=quantity,
        price=total_cost,
        date=transaction_date
    )

    delivery = get_supplier_delivery_date(transaction_date, quantity)

    return (
        f"Supplier order placed: {quantity} units of '{item_name}'. "
        f"Total cost: ${total_cost:.2f}. Expected delivery date: {delivery}."
    )

# Set up your agents and create an orchestration agent that will manage them.

# Four agents are used: one for inventory management, one for quoting, one for sales 
# transactions, and one for procurement.
class InventoryIntelligenceAgent(ToolCallingAgent):

    """
    Worker agent for real-time inventory assessment, saftey stock evaluation, and 
    identification of alternative in-stock items.
    """

    def __init__(self, model: OpenAIServerModel):
        
        super().__init__(
            tools=[check_all_inventory, check_stock_level, evaluate_reorder_needs],
            model=model,
            name="inventory_intelligence_agent",
            max_steps=10,
            description=(
                "Specialist agent for inventory assessment. "
                "Uses check_all_inventory to see all available stock. "
                "Uses check_stock_level to verify a specific item. "
                "Uses evaluate_reorder_needs to identify low or out-of-stock items and well-stocked alternatives. "
                "When a requested item is unavailable, identify the closest available alternative."
            ),
        )

class QuotingAgent(ToolCallingAgent):

    """
    Worker agent for generating competitive, customer-friendly quotes using historical
    pricing data and bulk discount logic.
    """

    def __init__(self, model: OpenAIServerModel):

        super().__init__(
            tools=[search_past_quotes, get_financial_summary, check_stock_level, get_catalog_price],
            model=model,
            name="quoting_agent",
            max_steps=4,
            description=(
                "Specialist agent for generating itemized customer quotes. "
                "ALWAYS use get_catalog_price to retrieve the correct base unit price for each item before quoting."
                "Uses search_past_quotes only as supplementary pricing context, not as the price source. "
                "Uses get_financial_summary to inform discount strategy. "
                "Uses check_stock_level to verify availability before quoting. "
                "Apply bulk discounts: 5% for 100 - 499 units, 10% for 500 - 999, and 15% for 1000+ units. "
                "Always include per-unit price, quantity, total, and discount rationale. "
                "If no historical quote is found, use the catalog unit price from check_stock_level or get_catalog_price as the base price. "
                "Never leave an item unpriced — use $0.10/unit as an absolute last-resort fallback. "
                "Never reveal internal cost prices, profit margins, or raw system errors."
            ),
        )

class SalesAgent(ToolCallingAgent):

    """
    Worker agent for finalizing sales transactions against available inventory.
    Handles only transaction_type = 'sales' (revenue-generating transactions).
    """

    def __init__(self, model: OpenAIServerModel):

        super().__init__(
            tools=[verify_stock_for_sale, record_sale],
            model=model,
            name="sales_agent",
            max_steps=6,
            description=(
                "Specialist agent for recording completed sales transactions. "
                "Process items ONE AT A TIME in strict sequence - never call multiple tools in the same step:\n"
                "  1. Call verify_stock_for_sale for one item. Wait for the result.\n"
                "  2. If stock is confirmed, call record_sale for that item. Wait for the result.\n"
                "  3. Move to the next item and repeat steps 1-2.\n"
                "  4. Only after ALL items have been processed, call final_answer ONCE with a summary.\n"
                "CRITICAL: Never call record_sale and final_answer in the same step. "
                "If record_sale returns 'ALREADY RECORDED', that item is completed - do not call it again. "
                "If stock is insufficient for any item, note the shortfall and continue to the next item."
            ),
        )

class ProcurementAgent(ToolCallingAgent):

    """
    Worker agent for managing supplier replenishment orders and delivery timelines.
    Handles only transaction_type = 'stock_orders' (cost-generating transactions).
    """

    def __init__(self, model: OpenAIServerModel):

        super().__init__(
            tools=[assess_restock_urgency, get_delivery_estimate, place_supplier_order],
            model=model,
            name="procurement_agent",
            max_steps=10,
            description=(
                "Specialist agent for supplier procurement and inventory replenishment. "
                "Uses assess_restock_urgency to evaluate how critical a restock is. "
                "Uses get_delivery_estimate to calculate when items will arrive. "
                "Uses place_supplier_order to record the purchase - pricing is handled automatically. "                
                "Always report the expected delivery date. "
                "Do not reveal supplier cost prices to customers."
            ),
        )


# Orchestrator agent to manage the workflow between the different worker agents and handle incoming requests
class MunderDifflinOrchestrator(ToolCallingAgent):

    """
    Main orchestrator for the Munder Difflin paper company multi-agent system.
    Coordinates InventoryIntelligenceAgent, QuotingAgent, SalesAgent, and ProcurementAgent to 
    handle customer order requests end-to-end.
    """

    def __init__(self, model: OpenAIServerModel):

        self.inventory_agent = InventoryIntelligenceAgent(model)
        self.quoting_agent = QuotingAgent(model)
        self.sales_agent = SalesAgent(model)
        self.procurement_agent = ProcurementAgent(model)

        @tool
        def handle_customer_request(request: str) -> str:

            """
            Process a customer inquiry through the full order lifecycle:
            inventory assessment, quote generation, sale or procurement.
            
            Args:
                request: Full customer request text including items, quantities, and date.
                
            Returns:
                Complete customer-facing response with quote, availability, and order status.
            """

            # Step 1 - Inventory: check stock, saftey levels, alternatives
            try:
                inventory_result = self.inventory_agent.run(
                    f"Customer request: '{request}'\n"
                    f"Complete these steps in order - do not repeat any step:\n"
                    f"1. Call check_all_inventory ONCE for the request date to get the full stock list.\n"
                    f"2. For each item mentioned in the request, call check_stock_level to confirm quantity.\n"
                    f"3. Call evaluate_reorder_needs ONCE to flag low stock items.\n"
                    f"4. If a requested item is out of stock, identify ONE alternative from the inventory list already retrieved in Step 1.\n"
                    f"Return your findings and stop - do not repeat tool calls."
                )
            except Exception as e:
                inventory_result = f"Inventory check unavailable: {e}"

            # Step 2 - Quoting: historical pricing + bulk discounts
            try:
                quote_result = self.quoting_agent.run(
                    f"Customer request: '{request}'\n"
                    f"Inventory assessment: (includes catalog unit prices per item: {inventory_result}\n"
                    f"1. Use the catalog unit prices from the invenntory assessment above as your base prices.\n"
                    f"  Only call get_catalog_price if a price was not returned by the inventory assessment.\n"
                    f"2. Once you have a price for each item - whether from catalog, get_catalog_price, or the $0.10 default - "
                    f"  do NOT look it up again. Generate the final itemized quote immediately. "
                    f"Quote for the ORIGINAL requested items using those prices. "
                    f"If an alternative was suggested in the inventory assessment, add it as a separate line marked 'Alternative option'. "
                    f"Do not re-check any price you have already retrieved.\n"
                    f"3. Search past quotes using search_past_quotes for supplementary context only.\n"
                    f"4. Generate an itemized quote: base catalog prices * quantity, then apply bulk discount.\n"
                    f"5. Apply bulk discounts (5% for 100 - 499 units, 10% for 500 - 999, 15% for 1000+).\n"
                    f"6. Use get_financial_summary to confirm discount capacity.\n"
                    f"Show per-unit price, discount applied, and line total. Do not reveal cost prices or margins."
                )
            except Exception as e:
                quote_result = f"Quote unavailable: {e}"

            # Step 3a - Sales: record sale for in-stock items
            try:
                sales_result = self.sales_agent.run(
                    f"Customer request: '{request}'\n"
                    f"Inventory assessment: {inventory_result}\n"
                    f"Quote: {quote_result}\n"
                    f"For each item with confirmed available stock:\n"
                    f"1. Use verify_stock_for_sale to confirm quantity.\n"
                    f"2. Use record_sale with the total quoted price for that item.\n"
                    f"For items with insufficient stock, clearly note they cannot be fulfilled now."
                )
            except Exception as e:
                sales_result = f"Sales processing unavailable: {e}"

            # Step 3b - Procurement: restock out-of-stock items
            try:
                procurement_result = self.procurement_agent.run(
                    f"Customer request: '{request}'\n"
                    f"Inventory assessment: {inventory_result}\n"
                    f"Sales result: {sales_result}\n"
                    f"For items that could NOT be fulfilled due to insufficient stock:\n"
                    f"1. Use assess_restock_urgency to evaluate how critical the restock is.\n"
                    f"2. Use get_delivery_estimate for the expected arrival date.\n"
                    f"3. Use place_supplier_order with item_name, quantity, and transaction_date - pricing is calculated automatically.\n"
                    f"Report: what was restocked and when it will arrive."
                )
            except Exception as e:
                procurement_result = f"Procurement unavailable: {e}"

            return (
                f"Thank you for your inquiry with Munder Difflin!\n\n"
                f"--- QUOTE ---\n{quote_result}\n\n"
                f"--- ORDER STATUS ---\n{sales_result}\n\n"
                f"--- FULFILLMENT & DELIVERY ---\n{procurement_result}\n\n"
            )
        
        super().__init__(
            tools=[handle_customer_request],
            model=model,
            name="munder_difflin_orchestrator",
            description=(
                "Main orchestrator for Munder Difflin paper company. "
                "Use handle_customer_request to process any customer inquiry end-to-end."
            ),
        )

    def process_request(self, request: str) -> str:
        
        """Process a customer request and return a complete message."""
        
        return self.run(
            f"Process this customer order request using handle_customer_request: '{request}'"
        )

# Run your test scenarios by writing them here. Make sure to keep track of them.

def run_test_scenarios():
    
    print("Initializing Database...")
    init_database(db_engine)
    try:
        quote_requests_sample = pd.read_csv("quote_requests_sample.csv")
        quote_requests_sample["request_date"] = pd.to_datetime(
            quote_requests_sample["request_date"], format="%m/%d/%y", errors="coerce"
        )
        quote_requests_sample.dropna(subset=["request_date"], inplace=True)
        quote_requests_sample = quote_requests_sample.sort_values("request_date")
    except Exception as e:
        print(f"FATAL: Error loading test data: {e}")
        return

    # Get initial state
    initial_date = quote_requests_sample["request_date"].min().strftime("%Y-%m-%d")
    report = generate_financial_report(initial_date)
    current_cash = report["cash_balance"]
    current_inventory = report["inventory_value"]

    ############
    ############
    ############
    # INITIALIZE YOUR MULTI AGENT SYSTEM HERE
    ############
    ############
    ############

    orchestrator = MunderDifflinOrchestrator(model)

    results = []
    for idx, row in quote_requests_sample.iterrows():
        request_date = row["request_date"].strftime("%Y-%m-%d")

        print(f"\n=== Request {idx+1} ===")
        print(f"Context: {row['job']} organizing {row['event']}")
        print(f"Request Date: {request_date}")
        print(f"Cash Balance: ${current_cash:.2f}")
        print(f"Inventory Value: ${current_inventory:.2f}")

        # Process request
        request_with_date = f"{row['request']} (Date of request: {request_date})"

        ############
        ############
        ############
        # USE YOUR MULTI AGENT SYSTEM TO HANDLE THE REQUEST
        ############
        ############
        ############

        # Implement error handling for orchestrator responseto catch any unexpected issues during processing and log 
        # them without crashing the system
        try:
            response = orchestrator.process_request(request_with_date)
        except Exception as e:
            response = "We were unable to process your request at this time. Please contact us directly."
            print(f"[Internal error on request {idx+1}: {e}]")

        # Update state
        report = generate_financial_report(request_date)
        current_cash = report["cash_balance"]
        current_inventory = report["inventory_value"]

        print(f"Response: {response}")
        print(f"Updated Cash: ${current_cash:.2f}")
        print(f"Updated Inventory: ${current_inventory:.2f}")

        results.append(
            {
                "request_id": idx + 1,
                "request_date": request_date,
                "cash_balance": current_cash,
                "inventory_value": current_inventory,
                "response": response,
            }
        )

        time.sleep(1)

    # Final report
    final_date = quote_requests_sample["request_date"].max().strftime("%Y-%m-%d")
    final_report = generate_financial_report(final_date)
    print("\n===== FINAL FINANCIAL REPORT =====")
    print(f"Final Cash: ${final_report['cash_balance']:.2f}")
    print(f"Final Inventory: ${final_report['inventory_value']:.2f}")

    # Save results
    pd.DataFrame(results).to_csv("test_results.csv", index=False)
    return results


if __name__ == "__main__":
    results = run_test_scenarios()

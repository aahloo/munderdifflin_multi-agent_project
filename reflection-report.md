# Reflection Report: Munder Difflin Multi-Agent Order System

---

## 1. Agent Workflow Architecture

The system is implemented as a five-agent hierarchy using the `smolagents` framework — one orchestrator and four specialized workers, each with a non-overlapping responsibility:

```
Customer Request
       ↓
 MunderDifflinOrchestrator
   ├──→ InventoryIntelligenceAgent   (Step 1: stock assessment & alternatives)
   ├──→ QuotingAgent                 (Step 2: historical pricing & bulk discounts)
   ├──→ SalesAgent                   (Step 3a: record fulfilled sales)
   └──→ ProcurementAgent             (Step 3b: restock out-of-stock items)
```

| Agent | Responsibility | Key Helper Functions |
|-------|---------------|----------------------|
| MunderDifflinOrchestrator | Sequences all workers; composes final response | — |
| InventoryIntelligenceAgent | Checks stock; flags low levels; suggests alternatives | `get_all_inventory`, `get_stock_level` |
| QuotingAgent | Historical pricing; bulk discounts; itemized quotes | `search_quote_history`, `get_cash_balance`, `generate_financial_report`, `get_stock_level` |
| SalesAgent | Verifies stock; records revenue-generating transactions | `get_stock_level`, `create_transaction` (`'sales'`) |
| ProcurementAgent | Urgency assessment; delivery estimation; supplier orders | `get_stock_level`, `get_supplier_delivery_date`, `create_transaction` (`'stock_orders'`) |

**Why 5 agents?** 
Each worker maps to one business function — inventory, pricing, sales, or purchasing — mirroring the real-world separation of departments with distinct KPIs and approval authorities.

**Why are SalesAgent and ProcurementAgent separate?** 
Both call `create_transaction`, but with opposite financial semantics: `'sales'` reduces inventory and increases cash; `'stock_orders'` increases inventory and decreases cash. Keeping them separate enforces clear financial accountability and prevents a single agent from both selling and restocking the same item in one step.

**Why Inventory → Quote → Sale/Procurement?** 
Running the inventory check first grounds every quote and transaction in real-time stock data, preventing the system from committing prices or sales against items that may have sold out earlier in the batch.

---

## 2. Evaluation Results

All 20 requests from `quote_requests_sample.csv` (April 1–17, 2025) were processed against a fully reset database.

| Metric | Value |
|--------|-------|
| Requests processed | 20 of 20 |
| Starting net cash | ~$45,059.70 |
| Final cash | $42,323.83 (−$2,736) |
| Final inventory value | $4,627.85 |
| Orders fulfilled | 10 of 20 (50%) |
| Out-of-stock / procurement responses | 19 of 20 (95%) |

**Strengths** 
Catalog-matched items (e.g., Cardstock at $0.15/unit, A4 paper at $0.05/unit) were priced correctly and bulk discounts (5%/10%/15%) applied consistently. The ProcurementAgent reported specific delivery dates for every unfulfilled order. Pipeline resilience was maintained via `try/except` isolation around each worker agent call, and an idempotency guard in `record_sale` prevented duplicate transactions when the LLM framework retried tool calls.

**Observed weakness — $0.10 default pricing** 
Customer requests used descriptive names (e.g., "A4 glossy paper," "heavy cardstock (white)") that did not exactly match the catalog's canonical names ("Glossy paper," "Cardstock"). Because `get_stock_level` uses an exact SQL match, these items returned zero stock even when equivalent items were available. To prevent a zero-price edge case from causing an infinite SalesAgent retry loop — a bug that produced −$6.4 million in net cash in early testing — a $0.10/unit fallback price was introduced. This value was chosen pragmatically (non-zero, within the low end of the catalog range) but was not derived from any business rule. Since procurement costs are calculated at 60% of real catalog prices, the $0.10 sale revenue was sometimes less than the restock cost, producing the observed net cash decline of ~$2,736 across the run.

---

## 3. Suggestions for Further Improvement

**Improvement 1 — Fuzzy catalog name matching** 
By replacing the exact SQL name lookup with TF-IDF vector scores (term frequency X inverse document frequency product), one could then measure the cosine similarity of both the requested and catalog vectors, and return the one with the highest score. When an exact match fails, the system would find the closest catalog equivalent above a similarity threshold and use its real price. For example, "A4 glossy paper" would resolve to "Glossy paper" at $0.20/unit, eliminating both the $0.10 fallback and the resulting cash drain. This method was actually implemented in the code for multi-agent RAG exercise in Lesson 7 (from sklearn.feature_extraction.text import TfidfVectorizer;
from sklearn.metrics.pairwise import cosine_similarity).

**Improvement 2 — Proactive inventory replenishment** 
The current system triggers procurement only after a customer request exposes a stock gap, meaning the customer receives an "out of stock" response on first contact. A proactive monitoring step — run before each batch of requests using `evaluate_reorder_needs` — would automatically restock any item below its `min_stock_level` before customer requests are processed, increasing the first-contact fulfillment rate and reducing the proportion of unfulfilled orders.

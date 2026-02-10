# Shadow Lead Time & Unrecorded Delay Cost Analysis

**Report Date:** February 10, 2026  
**Dataset:** Procurement KPI Analysis Dataset  
**Methodology:** RLM-Scheme automated analysis pipeline  

---

## Executive Summary

Out of **777 total purchase orders**, **560 were marked as Delivered**. Of those, **68 orders (12.1%)** have **no recorded Delivery Date** despite being marked as delivered — indicating **unrecorded delays** that mask true supplier performance and generate hidden inventory holding costs.

**Alpha_Inc** is the supplier costing the company the most through unrecorded delays, with an estimated **$4,610.23** in hidden inventory holding costs across **18 unrecorded orders** worth over **$1M** in total value.

The **grand total estimated holding cost** across all suppliers from unrecorded delays is **$16,034.19** annually at a 15% holding rate.

---

## Methodology

### Shadow Lead Time Definition
"Shadow Lead Time" refers to the estimated actual lead time for orders where the delivery date was never recorded. Since these orders were marked "Delivered" but lack a `Delivery_Date`, the true lead time is unknown — hence the "shadow."

### Estimation Approach
- **Official Lead Time** is calculated from all Delivered orders where both `Order_Date` and `Delivery_Date` exist.
- **Shadow Lead Time** is estimated as `max(avg_LT + 2×std_dev, max_observed_LT)` — a conservative 95th-percentile proxy reflecting the assumption that unrecorded deliveries are more likely to be delayed (recording failures correlate with process breakdowns).
- **Excess Days** = Shadow Lead Time − Average Official Lead Time.
- **Holding Cost** = Order Value × (Excess Days ÷ 365) × 15% annual holding rate.

---

## Supplier Rankings by Hidden Cost Impact

| Rank | Supplier | Unrecorded Orders | % of Their Deliveries | Avg Official LT (days) | Shadow LT Est. (days) | Excess Days | Total Order Value | Est. Holding Cost |
|------|----------|------------------:|----------------------:|------------------------:|----------------------:|------------:|------------------:|------------------:|
| 1 | **Alpha_Inc** | 18 | 16.8% | 10.7 | 21.9 | 11.2 | $1,001,627.79 | **$4,610.23** |
| 2 | Delta_Logistics | 17 | 14.2% | 10.3 | 22.5 | 12.2 | $762,721.04 | **$3,824.05** |
| 3 | Beta_Supplies | 10 | 9.1% | 11.2 | 23.2 | 12.0 | $571,018.36 | **$2,815.98** |
| 4 | Epsilon_Group | 13 | 10.8% | 10.5 | 21.7 | 11.2 | $605,270.74 | **$2,785.90** |
| 5 | Gamma_Co | 10 | 9.7% | 9.9 | 20.7 | 10.8 | $450,172.21 | **$1,998.02** |
| | **TOTAL** | **68** | **12.1%** | | | | **$3,390,810.14** | **$16,034.19** |

---

## Detailed Supplier Analysis

### 1. Alpha_Inc — Highest Cost Impact

- **Unrecorded Orders:** 18 (16.8% of their delivered orders — worst rate by volume)
- **Official Lead Time:** Avg 10.7 days | Median 11.0 days | Range 1–20 days | Std Dev 5.6 days
- **Shadow Lead Time Estimate:** 21.9 days (11.2 excess days per order)
- **Total Value at Risk:** $1,001,627.79
- **Estimated Holding Cost:** $4,610.23

**Top 5 Most Costly Unrecorded Orders:**

| PO_ID | Order Date | Category | Qty | Price | Order Value | Holding Cost |
|-------|-----------|----------|----:|------:|------------:|-------------:|
| PO-00369 | 2023-09-29 | Packaging | 1,732 | $71.71 | $124,201.72 | $571.67 |
| PO-00583 | 2022-12-10 | Raw Materials | 1,743 | $65.88 | $114,828.84 | $528.53 |
| PO-00348 | 2023-11-05 | Office Supplies | 1,684 | $62.53 | $105,300.52 | $484.67 |
| PO-00243 | 2022-10-13 | Raw Materials | 1,818 | $57.13 | $103,862.34 | $478.05 |
| PO-00090 | 2022-09-08 | Packaging | 1,071 | $93.91 | $100,577.61 | $462.93 |

### 2. Delta_Logistics

- **Unrecorded Orders:** 17 (14.2% of their delivered orders)
- **Official Lead Time:** Avg 10.3 days | Median 10.0 days | Range 1–20 days | Std Dev 6.1 days
- **Shadow Lead Time Estimate:** 22.5 days (12.2 excess days — highest excess)
- **Total Value at Risk:** $762,721.04
- **Estimated Holding Cost:** $3,824.05

**Top 5 Most Costly Unrecorded Orders:**

| PO_ID | Order Date | Category | Qty | Price | Order Value | Holding Cost |
|-------|-----------|----------|----:|------:|------------:|-------------:|
| PO-00648 | 2023-03-27 | Office Supplies | 1,685 | $75.21 | $126,728.85 | $635.38 |
| PO-00237 | 2022-03-11 | MRO | 1,458 | $81.56 | $118,914.48 | $596.20 |
| PO-00335 | 2022-03-11 | Raw Materials | 1,893 | $52.35 | $99,098.55 | $496.85 |
| PO-00597 | 2023-04-12 | MRO | 1,200 | $64.63 | $77,556.00 | $388.84 |
| PO-00057 | 2023-10-06 | Packaging | 1,578 | $42.77 | $67,491.06 | $338.38 |

### 3. Beta_Supplies

- **Unrecorded Orders:** 10 (9.1% of their delivered orders — best rate)
- **Official Lead Time:** Avg 11.2 days | Median 12.0 days | Range 1–20 days | Std Dev 6.0 days
- **Shadow Lead Time Estimate:** 23.2 days (12.0 excess days)
- **Total Value at Risk:** $571,018.36
- **Estimated Holding Cost:** $2,815.98

**Top 3 Most Costly Unrecorded Orders:**

| PO_ID | Order Date | Category | Qty | Price | Order Value | Holding Cost |
|-------|-----------|----------|----:|------:|------------:|-------------:|
| PO-00110 | 2022-11-19 | Office Supplies | 1,800 | $96.27 | $173,286.00 | $854.56 |
| PO-00014 | 2022-02-02 | MRO | 5,000 | $16.88 | $84,400.00 | $416.22 |
| PO-00099 | 2023-12-03 | Office Supplies | 779 | $90.92 | $70,826.68 | $349.28 |

### 4. Epsilon_Group

- **Unrecorded Orders:** 13 (10.8% of their delivered orders)
- **Official Lead Time:** Avg 10.5 days | Median 11.0 days | Range 1–20 days | Std Dev 5.6 days
- **Shadow Lead Time Estimate:** 21.7 days (11.2 excess days)
- **Total Value at Risk:** $605,270.74
- **Estimated Holding Cost:** $2,785.90

**Top 3 Most Costly Unrecorded Orders:**

| PO_ID | Order Date | Category | Qty | Price | Order Value | Holding Cost |
|-------|-----------|----------|----:|------:|------------:|-------------:|
| PO-00651 | 2022-10-20 | MRO | 1,807 | $60.00 | $108,420.00 | $499.03 |
| PO-00207 | 2022-08-05 | MRO | 1,244 | $84.78 | $105,466.32 | $485.43 |
| PO-00030 | 2023-08-27 | MRO | 1,005 | $73.31 | $73,676.55 | $339.11 |

### 5. Gamma_Co — Lowest Cost Impact

- **Unrecorded Orders:** 10 (9.7% of their delivered orders)
- **Official Lead Time:** Avg 9.9 days | Median 10.0 days | Range 1–20 days | Std Dev 5.4 days
- **Shadow Lead Time Estimate:** 20.7 days (10.8 excess days — lowest excess)
- **Total Value at Risk:** $450,172.21
- **Estimated Holding Cost:** $1,998.02

---

## Official vs Shadow Lead Time Comparison

| Supplier | Official Avg LT | Official Median LT | Official Max LT | Std Dev | Shadow LT Est. | Excess Days | Multiplier |
|----------|----------------:|--------------------:|-----------------:|--------:|----------------:|------------:|-----------:|
| Alpha_Inc | 10.7 | 11.0 | 20 | 5.6 | 21.9 | 11.2 | 2.05× |
| Delta_Logistics | 10.3 | 10.0 | 20 | 6.1 | 22.5 | 12.2 | 2.18× |
| Beta_Supplies | 11.2 | 12.0 | 20 | 6.0 | 23.2 | 12.0 | 2.07× |
| Epsilon_Group | 10.5 | 11.0 | 20 | 5.6 | 21.7 | 11.2 | 2.07× |
| Gamma_Co | 9.9 | 10.0 | 20 | 5.4 | 20.7 | 10.8 | 2.09× |

Shadow lead times are approximately **2× the official average** across all suppliers, indicating that unrecorded deliveries likely took roughly twice as long as the supplier's typical performance.

---

## Key Findings

1. **Alpha_Inc is the worst offender** — 18 unrecorded orders (highest volume) with the highest unrecorded-to-delivered ratio (16.8%) and the largest holding cost impact ($4,610.23). Their $1M+ in unrecorded order value is nearly double the next supplier.

2. **Delta_Logistics has the highest per-order excess** at 12.2 days, suggesting their unrecorded deliveries are the most severely delayed relative to their baseline performance.

3. **Beta_Supplies has the fewest unrecorded orders** (10, 9.1%) but their Shadow Lead Time estimate is the highest at 23.2 days, driven by higher variability in their official lead times (Std Dev 6.0).

4. **12.1% of all delivered orders lack delivery dates** — this is a systemic data quality issue, not isolated to one supplier.

5. **$3.39M in order value** is affected by missing delivery records, generating an estimated **$16,034.19 in hidden holding costs** annually.

---

## Recommendations

1. **Mandate delivery date recording** — Implement system controls that prevent marking orders as "Delivered" without entering a `Delivery_Date`.

2. **Prioritize Alpha_Inc for audit** — With the highest volume of unrecorded orders and cost impact, Alpha_Inc should be the first supplier reviewed for delivery performance compliance.

3. **Investigate Delta_Logistics lead time variability** — Their 12.2-day excess suggests systemic delays that are being obscured by missing data.

4. **Retroactively collect delivery dates** — For the 68 affected orders, work with suppliers and logistics to backfill actual delivery dates to get true cost impact.

5. **Include Shadow Lead Time as a KPI** — Track the ratio of unrecorded deliveries per supplier as a procurement quality metric in scorecards.

---

*Analysis performed using RLM-Scheme orchestration engine with Python data bridge. Holding cost calculated at 15% annual rate applied to order value over estimated excess lead time days.*

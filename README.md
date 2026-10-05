# Mistral LLM Benchmark for Multilingual Banking Intent Classification

A controlled benchmark for evaluating Mistral language models on multilingual banking intent classification, with a focus on Arabic, Egyptian Arabizi, English, and mixed-language banking queries.

The benchmark evaluates how effectively different Mistral models distinguish between customer-specific requests, general banking FAQs, hybrid requests, and out-of-scope queries under a consistent prompting and evaluation framework.

---

## Overview

Large Language Models (LLMs) are increasingly being used as intelligent routing components in banking and financial applications. Before deploying an LLM as an intent router, it is important to evaluate its ability to correctly identify different types of banking requests across languages and linguistic variations.

This repository provides a reproducible evaluation pipeline for Mistral models using the same:

- Dataset
- Classification taxonomy
- System prompt
- Generation parameters
- Evaluation methodology
- Output format

This controlled setup enables fair comparison between models while minimizing experimental variability.

---

## Research Objective

The primary objective of this benchmark is to evaluate Mistral models on multilingual banking intent classification.

The benchmark investigates:

- Overall classification accuracy
- Precision, recall, and F1-score
- Macro-averaged performance
- Performance across different linguistic conditions
- Model response latency
- Classification errors and failure patterns
- Differences between Mistral model families

Particular attention is given to Arabic and Egyptian Arabizi, which are comparatively underrepresented in many multilingual NLP benchmarks.

---

## Intent Taxonomy

Each banking query is classified into exactly one of four intent categories.

| Intent | Description |
|---|---|
| `customer` | Requests requiring customer-specific information, such as account balance, transactions, card details, or personal account information. |
| `faq` | General banking questions that can be answered without accessing customer-specific information. |
| `hybrid` | Requests that combine general banking information with customer-specific information or context. |
| `out_of_scope` | Requests unrelated to supported banking services or requests for unauthorized/private information. |

### Examples

| Query | Intent |
|---|---|
| `كم رصيد حسابي الجاري دلوقتي؟` | `customer` |
| `ما هي المستندات المطلوبة لفتح حساب جاري؟` | `faq` |
| `هل أستطيع رفع الحد اليومي لبطاقتي؟` | `hybrid` |
| `ما هو أفضل مطعم في القاهرة؟` | `out_of_scope` |

---

## Supported Languages

The benchmark covers several linguistic conditions relevant to real-world banking applications:

- Arabic
- Egyptian Arabic
- Egyptian Arabizi
- English
- Mixed Arabic-English queries

### Egyptian Arabizi

Egyptian Arabizi refers to Arabic written using Latin characters and numbers.

For example:

> ana 3ayez a3raf raseedy

Arabizi is treated as a distinct linguistic condition from conventional Arabic-English code-switching in order to evaluate model performance more precisely.

---

## Dataset

The benchmark uses a balanced dataset containing 100 banking queries across the four intent categories.

The dataset includes:

- Arabic banking queries
- Egyptian Arabic queries
- Egyptian Arabizi queries
- English queries
- Mixed-language queries
- Customer-specific requests
- General banking FAQs
- Hybrid requests
- Out-of-scope and adversarial queries

The benchmark dataset is stored in:

`dataset_balanced.csv`

### Dataset Structure

The dataset contains the query and corresponding ground-truth classification information.

Example:

```csv
id,query,language,route
1,"ما هي المستندات المطلوبة لفتح حساب جاري؟",arabic,faq
2,"كم رصيد حسابي الجاري دلوقتي؟",egyptian_arabic,customer
3,"ana 3ayez a3raf raseedy",arabizi,customer


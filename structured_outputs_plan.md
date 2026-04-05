# Hands-On Implementation Plan: Structured Output Types with Open Source LLMs

## Overview

This project demonstrates how Large Language Models (LLMs) can generate
**structured outputs** from unstructured text. The focus is on practical
implementations of different structured output formats used in modern AI
systems.

### Structured Output Types Covered

1.  JSON Output
2.  Key-Value Extraction
3.  Table Output
4.  Text-to-SQL Conversion
5.  Schema-Validated Output

------------------------------------------------------------------------

# 1. JSON Structured Output

## Goal

Convert unstructured text into JSON.

### Example Input

Invoice Number: 2345\
Vendor: Amazon\
Date: 12 Jan 2026\
Amount: 4500

### Example Code

``` python
from transformers import pipeline

generator = pipeline(
    "text-generation",
    model="google/flan-t5-large"
)

text = '''
Invoice Number: 2345
Vendor: Amazon
Date: 12 Jan 2026
Amount: 4500
'''

prompt = f"""
Extract structured data in JSON format.

Fields:
invoice_number
vendor
date
amount

Text:
{text}
"""

result = generator(prompt, max_length=200)
print(result[0]["generated_text"])
```

### Example Output

``` json
{
 "invoice_number": "2345",
 "vendor": "Amazon",
 "date": "12 Jan 2026",
 "amount": 4500
}
```

------------------------------------------------------------------------

# 2. Key-Value Structured Output

## Example Input

Patient Name: John Smith\
Age: 34\
Diagnosis: Diabetes\
Hospital: Apollo

### Example Prompt

    Extract key-value pairs from the following text.

### Example Output

``` json
{
"Patient Name": "John Smith",
"Age": 34,
"Diagnosis": "Diabetes",
"Hospital": "Apollo"
}
```

------------------------------------------------------------------------

# 3. Table Structured Output

## Example Input

Product A price 20\
Product B price 30\
Product C price 40

### Example Prompt

    Convert the following text into a table with columns: Product | Price

### Example Output

  Product   Price
  --------- -------
  A         20
  B         30
  C         40

------------------------------------------------------------------------

# 4. Text-to-SQL Structured Output

## Example Question

Show total sales for vendor Amazon

### Example Prompt

    Convert the question into SQL.

    Table: invoices
    Columns:
    invoice_number
    vendor
    amount
    date

### Example Output

``` sql
SELECT SUM(amount)
FROM invoices
WHERE vendor = 'Amazon';
```

------------------------------------------------------------------------

# 5. Schema-Validated Output

Use **Pydantic** to enforce structure.

### Schema Definition

``` python
from pydantic import BaseModel

class Invoice(BaseModel):

    invoice_number: str
    vendor: str
    date: str
    amount: float
```

### Validation

``` python
import json

data = json.loads(llm_output)

invoice = Invoice(**data)

print(invoice)
```

If the structure does not match, validation will fail.

------------------------------------------------------------------------

# Final Demo Pipeline

``` python
def structured_output_demo(text):

    json_output = extract_json(text)

    validated = validate_schema(json_output)

    sql_query = text_to_sql("total sales for vendor Amazon")

    print("JSON:", json_output)
    print("Validated:", validated)
    print("SQL:", sql_query)
```

------------------------------------------------------------------------

# Summary

  Structured Output Type   Description
  ------------------------ ----------------------------------------------
  JSON                     Convert text to machine-readable JSON
  Key-Value                Extract labeled information
  Table                    Organize text into rows and columns
  SQL                      Convert natural language to database queries
  Schema Validation        Ensure output matches required format

------------------------------------------------------------------------

# Real World Applications

-   Invoice Processing Systems
-   Healthcare Record Extraction
-   Financial Document Analysis
-   Business Intelligence Queries
-   Document AI Platforms

"""Tripletex Tool-Use Agent — LLM with on-demand API knowledge retrieval.

Uses Gemini 3.1 Pro with function calling. Instead of stuffing all API
knowledge into the system prompt, the LLM calls get_api_guide(topic) to
retrieve detailed documentation on demand. This keeps the context window
lean and focused.

Flow:
1. LLM reads the task prompt
2. Calls get_api_guide("customer") etc. to learn exact field names
3. Makes API calls via tripletex_get/post/put/delete
4. Reads responses and adapts
5. Continues until task is done
"""

import asyncio
import base64
import json
import logging
import time
from datetime import date

import vertexai
from vertexai.generative_models import (
    FunctionDeclaration,
    GenerativeModel,
    Part,
    Tool,
)

import warnings
warnings.filterwarnings("ignore", message=".*REST async clients.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

from tripletex_client import TripletexClient
from learning import record_error, compile_template

logger = logging.getLogger(__name__)

MAX_TURNS = 25
DEADLINE_BUFFER = 15  # stop 15s before timeout — safer margin

# ── Tool definitions ─────────────────────────────────────────────────

_tripletex_get = FunctionDeclaration(
    name="tripletex_get",
    description="GET request to Tripletex API. Use for searching/listing entities. Response includes 'value' (single entity) or 'values' (list). Always use fields param to limit response size.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer, /employee/123, /ledger/account"},
            "params": {"type": "object", "description": "Query parameters as key-value pairs, e.g. {\"name\": \"Acme\", \"fields\": \"id,name\"}"},
        },
        "required": ["path"],
    },
)

_tripletex_post = FunctionDeclaration(
    name="tripletex_post",
    description="POST request to Tripletex API. Use for creating new entities. Returns the created entity with its ID.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer, /employee, /order"},
            "body": {"type": "object", "description": "Request body as JSON object"},
            "params": {"type": "object", "description": "Optional query parameters"},
        },
        "required": ["path", "body"],
    },
)

_tripletex_put = FunctionDeclaration(
    name="tripletex_put",
    description="PUT request to Tripletex API. Use for updating entities or triggering actions (/:invoice, /:payment, /:send, /:approve, /:deliver, /:reverse).",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path, e.g. /customer/123, /order/456/:invoice, /invoice/789/:payment"},
            "body": {"type": "object", "description": "Request body (for updates, include id and version)"},
            "params": {"type": "object", "description": "Query parameters (for actions like /:payment, use params for paymentDate, paidAmount etc.)"},
        },
        "required": ["path"],
    },
)

_tripletex_delete = FunctionDeclaration(
    name="tripletex_delete",
    description="DELETE request to Tripletex API. Use for deleting entities.",
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "API path with entity ID, e.g. /travelExpense/123"},
        },
        "required": ["path"],
    },
)

_get_api_guide = FunctionDeclaration(
    name="get_api_guide",
    description="Get detailed API documentation for a specific topic. Call this BEFORE making API calls you're unsure about. Topics: customer, employee, invoice, voucher, travel_expense, project, supplier, product, department, contact, payment, credit_note, reminder, send_invoice, timesheet, salary, employment, opening_balance, supplier_invoice, purchase_order, asset, bank_reconciliation, dimensions, fixed_price_project, update_entity, receipt_voucher, employment_contract_pdf, bank_reconciliation_csv, ledger_analysis, supplier_invoice_pdf, currency_exchange, ledger_correction, overdue_invoice_reminder",
    parameters={
        "type": "object",
        "properties": {
            "topic": {"type": "string", "description": "The topic to get documentation for"},
        },
        "required": ["topic"],
    },
)

_get_api_schema = FunctionDeclaration(
    name="get_api_schema",
    description="Get exact field names for a Tripletex entity. Use when you need to know valid field names (e.g. for GET ?fields= or POST body). Entities: Customer, Employee, Product, Department, Supplier, Contact, Order, OrderLine, Invoice, Voucher, Posting, TravelExpense, TravelExpenseCost, TravelDetails, Project, ProjectHourlyRate, SalaryTransaction, PurchaseOrder, PurchaseOrderline, BankReconciliation, Asset, AccountingDimensionName, AccountingDimensionValue, Employment",
    parameters={
        "type": "object",
        "properties": {
            "entity": {"type": "string", "description": "Entity name (PascalCase), e.g. 'ProjectHourlyRate', 'TravelExpenseCost'"},
        },
        "required": ["entity"],
    },
)

TOOLS = [Tool(function_declarations=[
    _tripletex_get, _tripletex_post, _tripletex_put, _tripletex_delete, _get_api_guide, _get_api_schema
])]

# Load field reference from OpenAPI spec
import pathlib as _pathlib
_FIELD_REF_PATH = _pathlib.Path(__file__).parent / "schemas" / "field_reference.json"
_FIELD_REF: dict = {}
try:
    _FIELD_REF = json.loads(_FIELD_REF_PATH.read_text())
except Exception:
    pass

# ── Slim system prompt ───────────────────────────────────────────────

SYSTEM_PROMPT = f"""\
You are an expert Tripletex accounting agent. Today is {date.today().isoformat()}.
You receive accounting tasks in Norwegian, English, German, French, Spanish, Portuguese, or Nynorsk.
Execute each task by making Tripletex API calls using the provided tools.

BEFORE MAKING ANY API CALL:
1. Call get_api_guide for the MAIN entity type in the task
2. Call get_api_schema for any entity you're unsure about field names
3. Plan ALL your write calls (POST/PUT/DELETE) before starting
4. GET calls are FREE — use them to explore and verify

LANGUAGE GLOSSARY:
faktura=invoice, kunde=customer, ansatt=employee, leverandør=supplier, bilag=voucher, konto=account, prosjekt=project, avdeling=department, produkt=product, reiseregning=travel expense, innbetaling=payment, kreditnota=credit note, purring=reminder, bankavstemming=bank reconciliation, åpningsbalanse=opening balance, anleggsmiddel=fixed asset, lønn=salary, innkjøpsordre=purchase order, kontaktperson=contact person, ansettelse=employment, bokfør=post/book, reverser=reverse, godkjenn=approve, lever=deliver, slett=delete, Rechnung=invoice, Kunde=customer, Mitarbeiter=employee, Lieferant=supplier, facture=invoice, client=customer, employé=employee, fournisseur=supplier, factura=invoice, cliente=customer, empleado=employee, proveedor=supplier

CRITICAL UNIVERSAL RULES:
- The sandbox starts EMPTY — create ALL prerequisites (customer, supplier, employee) before dependent entities
- Account numbers are NOT account IDs — always GET /ledger/account?number=X&fields=id first
- All IDs must be integers, not strings
- Amounts must be numbers, not strings
- Dates must be YYYY-MM-DD format
- When updating entities, include both 'id' and 'version' from the GET response
- EVERY data point in the prompt MUST end up in the API calls — missing a phone, org number, or address = lost points

STRATEGY:
1. Parse the task to understand what entity types are involved
2. Call get_api_guide ONCE for the main entity type — don't call it for every sub-entity
3. Do NOT call get_api_guide or get_api_schema more than ONCE per task. If you already called it, use the info you got.
4. Plan ALL your API calls upfront before making the first one. Don't explore — execute.
5. For tasks with PDF/image attachments: The file content is provided as base64 in the request. The LLM can read it directly. Extract EVERY piece of data: names, numbers, dates, amounts, account numbers, department names, salary details.
6. For analysis tasks: query existing data via GET endpoints before creating new entities
7. Create prerequisites first (customer before invoice, accounts before voucher)
8. Call MULTIPLE tools in a single turn when they are independent (e.g. GET department + GET employee can be parallel)
9. If a call fails, read the error and adapt — NEVER repeat the same failing call
10. When done, stop IMMEDIATELY — no verification calls, no summaries
11. After the last required write call, STOP. Do not verify, summarize, or make extra calls.

EFFICIENCY (you have 290 seconds total):
- GET requests are FREE — they don't count toward efficiency score. Read as much as you need!
- Only POST/PUT/DELETE count as "write calls" — minimize these
- Combine independent API calls in the same turn (parallel function calling)
- Use GET to verify data, understand structure, find existing entities — it's FREE
- Minimize write ERRORS (4xx on POST/PUT/DELETE) — each error reduces efficiency bonus
- Do NOT retry a failing write more than once — read the error, fix the issue, try once more

ENDPOINTS THAT DO NOT EXIST (cause 404/405 — NEVER use these):
- /travelExpense/ID/expenses, /travelExpense/ID/:addExpense, /travelExpense/rateType, /expense
- /orderline (orderLines go IN POST /order body, NOT as separate endpoint)
- POST /supplierInvoice ALWAYS returns 500 — NEVER use it! Use POST /ledger/voucher instead
- PUT /company/modules (returns 405)
- PUT /salary/payslip/ID (returns 405 — payslips are READ-ONLY after creation)
- PUT /salary/transaction/ID (returns 405 — transactions are READ-ONLY)
- POST /salary/transaction/line (returns 405)
- GET /salary/payslip/ID/line (returns 404)
- PUT /invoice/ID/:reverse (does NOT exist — use PUT /ledger/voucher/ID/:reverse instead)
- GET /invoice with field "totalAmountExcludingVatCurrency" or "description" (invalid fields)
- DELETE /employee/employment/ID (returns 405 — employments cannot be deleted)
- PUT /employee/employment/ID with "department" field (doesn't exist on employment — department is on employee)
- Some accounts are LOCKED to vatType 0: 1500, 1920, 2400, 3400, 7350, 8060, 8160 and other non-VAT accounts. If you get "Kontoen er låst til mva-kode 0", use vatType:{"id":0} for that posting
- ALWAYS use GET /ledger/account?number=X&fields=id,vatType to check if account has locked vatType BEFORE posting
- PUT /invoice/ID/:send MUST include sendType param (e.g. sendType=EMAIL)
- GET /currency: fields are id, code, description, displayName, factor (NOT name — causes 400)
- Employment lookup: GET /employee/employment?employeeId=ID (NOT /employee/ID/employment)

MANDATORY FIELD RULES (violating these = instant 422):
- Product: field is "number" (NOT productNumber, NOT productNo)
- Product vatType: must be {{"id": N}} where N = 3 (25%), 33 (15% food), 31 (12%), 5 (0%)
- Voucher: "description" is REQUIRED (not optional)
- Employee: phone is "phoneNumberMobile" (NOT phone, NOT phoneNumber, NOT mobileNumber)
- POST /supplierInvoice: BROKEN (always 500). SKIP it entirely — use POST /ledger/voucher directly
- POST /employee: MUST include userType:"STANDARD" AND department:{{"id":X}} (GET /department first!)
- POST /travelExpense: isDayTrip and isForeignTravel go INSIDE travelDetails (NOT top-level body)
- POST /travelExpense/cost: amountCurrencyIncVat is REQUIRED. costCategory must be {{"id":X}} object (NOT string)
- GET /invoice: MUST include invoiceDateFrom AND invoiceDateTo params (both required)
- PUT /:invoice: MUST include invoiceDueDate param (invoiceDate + 14 days if not specified)
- PUT /:reverse: date goes as QUERY param (not body)
- GET /ledger/voucher: MUST include dateFrom AND dateTo params (both required). dateTo must be AFTER dateFrom (not same day! use dateFrom=2026-03-20&dateTo=2026-03-21)
- Voucher postings: row starts from 1, MUST include amountGrossCurrency AND vatType
- ProjectHourlyRate: rate field is "fixedRate" (NOT hourlyRate). hourlyRateModel is a string like "TYPE_FIXED_HOURLY_RATE"
- orderLine.vatType MUST be an object {{"id": N}}, NOT a bare number. Common IDs: 3=25% outgoing, 33=15% food, 5=0% exempt
- NEVER PUT /activity — activities are read-only. Use GET /activity to find existing ones, don't try to modify them.
- POST /activity requires activityType field. But NEVER create activities — use GET /activity?isProjectActivity=true to find existing ones.
- /activityType endpoint does NOT exist (404). Activity types are predefined.
- If timesheet date < project startDate, PUT /project to change startDate (NOT PUT /activity)
"""

# ── API Guides (on-demand knowledge) ────────────────────────────────

API_GUIDES: dict[str, str] = {
    "customer": """\
## Customer
POST /customer {"name":"X", "isCustomer":true, "email":"x@y.no", "organizationNumber":"123456789", "phoneNumber":"12345678"}
- postalAddress: {"addressLine1":"X", "postalCode":"1234", "city":"Oslo"}
- physicalAddress: same structure as postalAddress
- For both customer AND supplier: add "isSupplier":true
- phoneNumber is the correct field (NOT phone, NOT phoneNumberMobile)
- ALWAYS include organizationNumber if mentioned in prompt
- ALWAYS include ALL data from prompt (email, phone, address, org number)
- If customer already exists (409 Conflict), GET /customer?name=X&fields=id to find existing
Example:
POST /customer {"name":"Acme AS", "isCustomer":true, "email":"post@acme.no", "organizationNumber":"987654321", "phoneNumber":"22334455", "postalAddress":{"addressLine1":"Storgata 1","postalCode":"0155","city":"Oslo"}}
""",

    "employee": """\
## Employee
POST /employee {"firstName":"X", "lastName":"Y", "email":"x@y.no", "dateOfBirth":"1990-01-01", "phoneNumberMobile":"99887766", "userType":"STANDARD", "department":{{"id":DEPT_ID}}}
- MUST GET /department?fields=id,name first and include department.id (required field!)
- Phone field is phoneNumberMobile (NOT phoneNumber, NOT mobileNumber — these cause 422)
- email field is immutable after creation
- If email "allerede i bruk": GET /employee?email=X&fields=id to find existing employee
- dateOfBirth: include if mentioned, format YYYY-MM-DD
- employeeNumber: auto-assigned, returned in response
Role/admin privileges:
- PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=ID&template=ALL_PRIVILEGES
- Templates: ALL_PRIVILEGES, INVOICING_MANAGER, PERSONELL_MANAGER, ACCOUNTANT, AUDITOR, DEPARTMENT_LEADER
""",

    "employment": """\
## Employment (ansettelse)
Step 1: POST /employee/employment {{"employee":{{"id":X}}, "startDate":"2026-01-01"}}
- ONLY employee.id and startDate — NO other fields
- Returns employment with ID

Step 2 (optional — salary/percentage/hours): POST /employee/employment/details {{
  "employment":{{"id":EMPLOYMENT_ID}},
  "date":"YYYY-MM-DD",
  "percentageOfFullTimeEquivalent":100,
  "annualSalary":500000,
  "occupationCode":{{"id":OCC_ID}}
}}
- Valid fields: employment, date, employmentType, employmentForm, remunerationType, workingHoursScheme, shiftDurationHours, occupationCode, percentageOfFullTimeEquivalent, annualSalary, hourlyWage, monthlySalary
- INVALID fields (cause 422): workingHoursPerWeek, hoursPerWeek, workingHours, fullTimeEquivalentPercentage, standardWorkingHoursPerWeek, salary, type
- GET /employee/employment/occupationCode?fields=id,code,nameNO for valid occupation codes
""",

    "invoice": """\
## Invoice (faktura) — Create via Order
1. POST /customer (if new) — see get_api_guide("customer")
2. POST /order {"customer":{{"id":X}}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{"description":"Item", "count":1, "unitPriceExcludingVatCurrency":1000, "vatType":{{"id":3}}}]}
   - BOTH orderDate AND deliveryDate are REQUIRED
   - deliveryDate defaults to orderDate if not specified in task
3. PUT /order/ORDER_ID/:invoice?sendToCustomer=false&invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD
   - ALWAYS include invoiceDueDate param — default: invoiceDate + 14 days
   - If invoicing fails with "bankkontonummer": POST /bank to register bank account first

With payment after invoicing:
- GET /invoice/paymentType?fields=id,description first
- PUT /invoice/INV_ID/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=AMOUNT
""",

    "voucher": """\
## Voucher (bilag)
1. GET /ledger/account?number=XXXX&fields=id for EACH account number
2. POST /ledger/voucher {"date":"YYYY-MM-DD", "description":"X", "postings":[
     {"row":1, "account":{"id":DEBIT_ACCT_ID}, "amountGross":AMOUNT, "amountGrossCurrency":AMOUNT, "vatType":{"id":VAT_ID}},
     {"row":2, "account":{"id":CREDIT_ACCT_ID}, "amountGross":-AMOUNT, "amountGrossCurrency":-AMOUNT, "vatType":{"id":VAT_ID}}
   ]}

CRITICAL RULES:
- Row starts from 1 (NEVER 0)
- MUST include amountGrossCurrency (same value as amountGross)
- MUST include vatType on every posting
- Postings MUST sum to zero (total debit = total credit)

vatType IDs:
- 0 = no VAT (1xxx, 2xxx, 5xxx, 8xxx accounts — balance sheet/equity)
- 1 = incoming 25% (4xxx, 6xxx, 7xxx accounts — expenses)
- 3 = outgoing 25% (3xxx accounts — revenue)

AMOUNT RULES: amountGross is always the GROSS amount (INCLUDING VAT). Tripletex automatically calculates the VAT split based on the vatType. You provide the FULL amount, Tripletex handles the rest.
""",

    "travel_expense": """\
## Travel Expense (reiseregning)
1. GET /employee or POST /employee (need employee.id)
2. POST /travelExpense {"title":"X", "employee":{{"id":X}}, "travelDetails":{"departureDate":"YYYY-MM-DD", "returnDate":"YYYY-MM-DD", "departureFrom":"Oslo", "destination":"Bergen", "purpose":"X", "isDayTrip":false, "isForeignTravel":false}}
   - isDayTrip and isForeignTravel go INSIDE travelDetails (NOT top-level!)
   - departureFrom, destination, purpose also go inside travelDetails
3. GET /travelExpense/costCategory?fields=id,description — find cost category IDs
4. GET /travelExpense/paymentType?fields=id,description — find payment type ID
5. For EACH cost: POST /travelExpense/cost {"travelExpense":{"id":TE_ID}, "date":"YYYY-MM-DD", "amountCurrencyIncVat":AMOUNT, "vatType":{"id":0}, "paymentType":{"id":PT_ID}, "costCategory":{"id":CAT_ID}}
   - amountCurrencyIncVat is the REQUIRED amount field (NOT costCurrency, NOT amount)
   - costCategory MUST be an object {{"id":X}} (NOT a string! "category":"Flight" causes 422)
   - Match category by description from GET /travelExpense/costCategory (e.g. find "Flyreise" for flights)
   - For per diem (diett/dieta): use POST /travelExpense/cost with amountCurrencyIncVat = daily_rate * days (e.g. 800kr/day * 4 days = 3200). Use a separate cost entry for per diem alongside other costs (flight, taxi etc.)

ENDPOINTS THAT DON'T EXIST:
- /travelExpense/ID/expenses, /travelExpense/ID/:addExpense, /travelExpense/rateType, /expense
- NEVER include "expenses" field in POST/PUT /travelExpense body — use /travelExpense/cost separately!

Actions:
- DELETE /travelExpense/ID — delete
- PUT /travelExpense/ID/:deliver — submit
- PUT /travelExpense/ID/:approve — approve
""",

    "project": """\
## Project (prosjekt)
1. GET /department?fields=id,name&count=1
2. POST /employee (project manager) with department.id
3. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=ID&template=ALL_PRIVILEGES
4. POST /customer (if external project)
5. POST /project {"name":"X", "startDate":"YYYY-MM-DD", "projectManager":{"id":EMP_ID}, "isInternal":false, "customer":{"id":CUST_ID}}
   - Internal project: set isInternal:true, omit customer
   - isFixedPrice: set to true for fixed-price projects, with fixedprice:AMOUNT

## Project Hourly Rates
GET /project/hourlyRates?projectId=PROJECT_ID&fields=id,fixedRate,hourlyRateModel,startDate,version
PUT /project/hourlyRates/ID {"id":X, "version":V, "fixedRate":1600, "hourlyRateModel":"TYPE_FIXED_HOURLY_RATE", "startDate":"YYYY-MM-DD"}
- Rate field is "fixedRate" (number) — NOT hourlyRate, rate, price, amount
- hourlyRateModel must be string: "TYPE_FIXED_HOURLY_RATE" (NOT an object, NOT a number)
- Valid fields: id, version, project, startDate, showInProjectOrder, hourlyRateModel, projectSpecificRates, fixedRate
- INVALID fields (cause 422): hourlyRate, hourlyRateCost, rate, price, amount, activity, employee

## Project Invoicing (fakturering basert på timer)
To invoice logged hours:
1. Register timesheet entries first (see get_api_guide("timesheet"))
2. POST /order {"customer":{{"id":X}}, "project":{"id":PROJ_ID}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{"description":"X", "count":HOURS, "unitPriceExcludingVatCurrency":HOURLY_RATE}]}
3. PUT /order/ORDER_ID/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD&sendToCustomer=false

For fixed-price partial invoicing (e.g. "75% av fastpris"):
1. Create project with isFixedPrice:true, fixedprice:TOTAL
2. POST /order with orderLines amount = fixedprice * percentage / 100
3. PUT /order/ORDER_ID/:invoice

## Reversed/cancelled payment (stornering/tilbakeføring)
To reverse a payment on an invoice:
1. Create customer, order, invoice, register payment (full invoice flow)
2. Find the voucher for the payment: GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&fields=id,date,description
   - MUST include dateFrom AND dateTo params (both required!)
3. PUT /ledger/voucher/VOUCHER_ID/:reverse?date=YYYY-MM-DD
   - MUST include date as query param (not body!)
4. This makes amountOutstanding on the invoice equal to the original amount again

Alternative: PUT /invoice/ID/:createCreditNote?date=YYYY-MM-DD for full reversal
""",

    "supplier": """\
## Supplier (leverandør)
POST /supplier {"name":"X", "organizationNumber":"123456789", "email":"x@y.no", "phoneNumber":"12345678"}
- postalAddress: {"addressLine1":"X", "postalCode":"1234", "city":"Oslo"}
- A customer can also be a supplier: POST /customer with "isSupplier":true
""",

    "product": """\
## Product (produkt)
POST /product {"name":"X", "number":"P001", "priceExcludingVatCurrency":1000}
- If number "er i bruk" (409): GET /product?number=X&fields=id to find existing product ID
- priceExcludingVatCurrency is the price field (NOT price, NOT unitPrice)
""",

    "department": """\
## Department (avdeling)
POST /department {"name":"X", "departmentNumber":123}
- departmentNumber is required and must be unique
- Returned in employee responses as department.id
""",

    "contact": """\
## Contact Person (kontaktperson)
POST /contact {"firstName":"X", "lastName":"Y", "email":"x@y.no", "customer":{{"id":X}}}
- Phone field is phoneNumberMobile (NOT phoneNumber — that field doesn't exist on contact)
- Must link to a customer via customer.id
""",

    "payment": """\
## Payment (innbetaling)
For invoice payment:
1. GET /invoice/paymentType?fields=id,description — get payment type ID
2. PUT /invoice/INV_ID/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=AMOUNT
   - All params go as query params, NOT body

To FIND an invoice:
- GET /invoice?invoiceDateFrom=2026-01-01&invoiceDateTo=2026-12-31&fields=id,invoiceNumber,amount,amountOutstanding,customer
- MUST include invoiceDateFrom AND invoiceDateTo (both required!)
- Valid fields: id, version, invoiceNumber, invoiceDate, invoiceDueDate, amount, amountOutstanding, amountCurrency, customer, kid, comment
- INVALID fields (cause 400): voucherNumber, amountExVat, totalAmount, status

To reverse/undo a payment:
1. Create the full invoice flow first (customer → order → invoice → payment)
2. GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&fields=id,date,description — find the payment voucher
3. PUT /ledger/voucher/VOUCHER_ID/:reverse?date=YYYY-MM-DD — reverse it (date is QUERY param!)
""",

    "credit_note": """\
## Credit Note (kreditnota)
Full flow (sandbox is empty — must create everything from scratch):
1. POST /customer {"name":"X", "isCustomer":true, "organizationNumber":"123456789"}
2. POST /order {"customer":{"id":CUST_ID}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{"description":"X", "count":1, "unitPriceExcludingVatCurrency":AMOUNT}]}
3. PUT /order/ORDER_ID/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD&sendToCustomer=false
4. PUT /invoice/INV_ID/:createCreditNote?date=YYYY-MM-DD&comment=Kreditering
   - date is REQUIRED as query param
   - comment is optional query param
   - Returns the credit note invoice object

IMPORTANT: The sandbox starts EMPTY. There is NO pre-existing invoice. You MUST create customer → order → invoice → credit note.
""",

    "reminder": """\
## Reminder (purring)
PUT /invoice/ID/:createReminder?dispatchType=EMAIL
- dispatchType=EMAIL (NOT sendType, NOT sendMethod)
- Dispatch types: EMAIL, EHF, EFAKTURA
""",

    "send_invoice": """\
## Send Invoice
PUT /invoice/ID/:send?sendType=EMAIL
- sendType UPPERCASE: EMAIL, EHF, EFAKTURA
""",

    "timesheet": """\
## Timesheet Entry (timeføring)
1. GET /employee or POST /employee (need employee.id)
2. GET /project?name=X&fields=id,name,startDate,version
3. GET /activity?isProjectActivity=true&fields=id,name (MUST use project activity, not general)
4. If timesheet date < project startDate: PUT /project to adjust startDate first
5. POST /timesheet/entry {"employee":{{"id":X}}, "project":{{"id":X}}, "activity":{{"id":X}}, "date":"YYYY-MM-DD", "hours":N, "comment":"X"}
   - FORBIDDEN fields: description, title, name, type (cause 422)
   - Use "comment" for any text description

## Timesheet Entry + Project Invoice (log hours then invoice customer)
1. POST /customer (create customer)
2. GET /department (for employee)
3. POST /employee (create named employee with department)
4. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=ID&template=ALL_PRIVILEGES
5. POST /project {{"name":"X", "startDate":"YYYY-MM-DD", "projectManager":{{"id":EMP_ID}}, "customer":{{"id":CUST_ID}}}}
6. GET /activity?isProjectActivity=true&fields=id,name (find project activity)
7. POST /timesheet/entry {{"employee":{{"id":X}}, "project":{{"id":X}}, "activity":{{"id":X}}, "date":"YYYY-MM-DD", "hours":N, "comment":"X"}}
8. POST /order {{"customer":{{"id":CUST_ID}}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{{"description":"X hours @ Y NOK/h", "count":1, "unitPriceExcludingVatCurrency": HOURS * HOURLY_RATE}}]}}
9. PUT /order/ORDER_ID/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD&sendToCustomer=false

IMPORTANT: The invoice amount = hours x hourlyRate. Calculate this from the prompt values.
FORBIDDEN on /timesheet/entry: description, title, name, type — use "comment" instead.
Activity MUST have isProjectActivity=true.
NEVER PUT /activity — activities are read-only (403 Forbidden). Only GET /activity to find existing ones.
If timesheet date < project startDate, PUT /project to change startDate (NOT PUT /activity).
""",

    "salary": """\
## Salary (lønn)
Steps to run payroll:
1. POST /employee (create the employee if needed, with dateOfBirth!)
2. POST /employee/employment {"employee":{{"id":X}}, "startDate":"YYYY-MM-DD"} (employment required for salary)
3. GET /salary/type?fields=id,number,name — find salary type IDs
   - Common types: number "2000" = Fastlønn (base salary), number "2001" = Timelønn
4. POST /salary/transaction {"date":"YYYY-MM-DD", "year":2026, "month":3, "payslips":[{"employee":{"id":EMP_ID}}]}
   - Returns transaction with payslip IDs
5. To add salary lines/specifications to the payslip, use the returned payslip data
   - The payslip has "specifications" array with salary details

FORBIDDEN fields on /salary/transaction: salaryLines, salaryTransaction, line, amount, baseSalary
FORBIDDEN endpoints: /salary/transaction/line (405), /salary/payslip/ID/line (404)

For bonus: create a second specification on the same payslip with a different salary type

Note: Salary requires employee to have dateOfBirth set AND an active employment record.
""",

    "opening_balance": """\
## Opening Balance (åpningsbalanse)
POST /ledger/voucher with description "Åpningsbalanse"
Each posting needs:
- row (1,2,3...), account.id, amountGross (positive=debit, negative=credit), amountGrossCurrency (same as amountGross), vatType.id
- Use vatType.id=0 for balance sheet accounts (1xxx/2xxx)
- Postings MUST sum to zero
- If only asset accounts given, add equity account 2050 as balancing entry
- GET /ledger/account?number=XXXX&fields=id for each account number first
""",

    "supplier_invoice": """\
## Supplier Invoice (leverandørfaktura)
1. POST /supplier {"name":"X", "organizationNumber":"X"} — create supplier
2. GET /ledger/account?number=EXPENSE_ACCT&fields=id (expense account, e.g. 6500, 6700, 7100)
3. GET /ledger/account?number=2400&fields=id (accounts payable)
4. POST /ledger/voucher (NEVER use /supplierInvoice — it always returns 500!)
   {"date":"YYYY-MM-DD", "description":"Leverandørfaktura X fra Y", "postings":[
       {"row":1, "account":{"id":EXPENSE_ID}, "amountGross":AMOUNT, "amountGrossCurrency":AMOUNT, "vatType":{"id":1}, "supplier":{"id":SUPPLIER_ID}},
       {"row":2, "account":{"id":AP_2400_ID}, "amountGross":-AMOUNT, "amountGrossCurrency":-AMOUNT, "vatType":{"id":0}, "supplier":{"id":SUPPLIER_ID}}
     ]}

CRITICAL: Each posting in the voucher MUST include supplier: {"id": SUPPLIER_ID}
Without supplier reference, you get "Leverandør mangler" error.

DO NOT include: orderDate, deliveryDate, dueDate (cause 422)
amountGross = GROSS amount (including VAT). Tripletex calculates VAT automatically.

AMOUNT RULES:
- amountGross = the FULL amount (including VAT if applicable)
- For expense posting (vatType 1/25%): amountGross = full amount. Tripletex calculates net and VAT automatically.
- For AP posting (vatType 0): amountGross = negative full amount
- Both postings use the SAME absolute amount (just positive/negative)
- Postings MUST sum to zero
- Example: 61200 TTC → expense: 61200, AP: -61200

vatType mapping:
- Expense accounts 4xxx,6xxx,7xxx → vatType:1 (incoming VAT 25%)
- AP account 2400 → vatType:0 (no VAT)
""",

    "purchase_order": """\
## Purchase Order (innkjøpsordre)
1. POST /supplier (if needed)
2. GET /employee?count=1&fields=id (for ourContact)
3. POST /purchaseOrder {"supplier":{{"id":X}}, "ourContact":{{"id":X}}, "deliveryDate":"YYYY-MM-DD"}
4. POST /purchaseOrder/orderline {"purchaseOrder":{"id":PO_ID}, "description":"X", "count":N, "unitPriceExcludingVatCurrency":AMOUNT}
   - orderLines CANNOT be included in POST /purchaseOrder body (causes "purchaseOrder: Kan ikke være null")
   - Must POST each orderline separately AFTER creating the purchase order
""",

    "asset": """\
## Fixed Asset (anleggsmiddel)
POST /asset {"name":"X", "dateOfAcquisition":"YYYY-MM-DD", "acquisitionCost":AMOUNT}
""",

    "bank_reconciliation": """\
## Bank Reconciliation (bankavstemming)
POST /bank/reconciliation {"account":{{"id":X}}, "type":"MANUAL", "dateFrom":"YYYY-MM-DD"}
- account.id is the ledger account ID (GET /ledger/account?number=1920&fields=id)
""",

    "dimensions": """\
## Accounting Dimensions (fri regnskapsdimensjon)
1. POST /ledger/accountingDimensionName {"dimensionName":"Kostsenter"} — creates dimension (field is dimensionName, NOT name)
   - If dimensionName "er i bruk": GET /ledger/accountingDimensionName?fields=id,dimensionName to find existing ID — use it instead of creating
2. POST /ledger/accountingDimensionValue {"displayName":"Økonomi", "dimensionIndex":1} — creates value
   - displayName is the value name (NOT name)
   - dimensionIndex: 1 for first free dimension, 2 for second, 3 for third
   - If dimensionValue "er i bruk": GET /ledger/accountingDimensionValue?fields=id,displayName to find existing ID — use it instead of creating
3. Then create a voucher with the dimension linked to a posting:
   - GET /ledger/account?number=XXXX&fields=id for the debit account
   - GET /ledger/account?number=1920&fields=id for the credit (bank) account
   - POST /ledger/voucher with postings:
     [{"row":1, "account":{"id":DEBIT_ID}, "amountGross":AMOUNT, "amountGrossCurrency":AMOUNT, "vatType":{"id":0}, "freeAccountingDimension1":{"id":DIM_VALUE_ID}},
      {"row":2, "account":{"id":BANK_ID}, "amountGross":-AMOUNT, "amountGrossCurrency":-AMOUNT, "vatType":{"id":0}}]
   - freeAccountingDimension1 links to the FIRST dimension value (NOT freeDimension1)
""",

    "fixed_price_project": """\
## Fixed-Price Project (fastprisprosjekt)
POST /project with isFixedPrice:true, fixedprice:AMOUNT
- For partial invoicing (e.g. "75% av fastpris"):
  1. Create project with isFixedPrice:true, fixedprice:TOTAL
  2. POST /order with customer, orderDate, deliveryDate, orderLines with computed amount (fixedPrice * percentage / 100)
  3. PUT /order/ID/:invoice
- projectManager must have ALL_PRIVILEGES entitlement
""",

    "update_entity": """\
## Updating Any Entity
PUT /entity/ID with body including "id" and "version" from the GET response.
1. GET /entity/ID?fields=id,version,... to get current version
2. PUT /entity/ID {"id":ID, "version":VERSION, ...updated fields...}
- version is required for optimistic locking — without it you get 409 Conflict
- Include all fields you want to keep (PUT replaces the entity)
""",

    "receipt_voucher": """\
## Receipt to Voucher (kvittering → bilag)
The task gives you a receipt image/PDF. You must:
1. Read the receipt — extract: vendor name, amount, date, what was purchased
2. Determine the correct expense account based on purchase type:
   - Hotel/overnatting → 7140 (Reise og diett)
   - Restaurant/mat → 7100 (Bilkostnader) or 6340 (Serveringskostnader)
   - Office supplies/kontor → 6540 (Inventar og utstyr)
   - Phone/telefon → 6900 (Telefon)
   - Transport/taxi → 7120 (Bilgodtgjørelse)
   - Parking → 7130 (Parkering)
   - Flight/fly → 7140 (Reise og diett)
3. Determine VAT: 25% standard, 15% food, 12% transport/hotel, 0% exempt
4. If department is specified: GET /department?name=X, include department ref
5. POST /ledger/voucher with postings (expense account debit, 1920 bank credit)
6. amountGross = the GROSS/FULL amount (including VAT). Tripletex calculates VAT split automatically based on vatType.

Steps:
1. GET /ledger/account?number=EXPENSE_ACCT&fields=id (e.g. 7140)
2. GET /ledger/account?number=1920&fields=id (bank account)
3. If department specified: GET /department?name=X&fields=id
4. POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Kvittering: vendor - description",
     "postings":[
       {{"row":1, "account":{{"id":EXPENSE_ID}}, "amountGross":FULL_AMOUNT, "amountGrossCurrency":FULL_AMOUNT, "vatType":{{"id":1}}, "department":{{"id":DEPT_ID}}}},
       {{"row":2, "account":{{"id":BANK_ID}}, "amountGross":-FULL_AMOUNT, "amountGrossCurrency":-FULL_AMOUNT, "vatType":{{"id":0}}}}
     ]}}

VAT type IDs:
- 1 = incoming 25% (standard goods/services)
- 11 = incoming 15% (food)
- 13 = incoming 12% (transport/hotel)
- 0 = no VAT (exempt)
""",

    "employment_contract_pdf": """\
## Employment Contract PDF → Create Employee
1. Read the PDF attachment — extract ALL fields:
   - Name (firstName, lastName)
   - personnummer/national ID (nationalIdentityNumber)
   - dateOfBirth
   - Email, phone
   - Department
   - Position/stillingskode
   - Salary/lønn
   - Start date
   - Employment percentage (percentageOfFullTimeEquivalent)
2. GET /department?name=X&fields=id or POST /department {{"name":"X", "departmentNumber":N}} if needed
3. POST /employee {{"firstName":"X", "lastName":"Y", "email":"x@y.no", "dateOfBirth":"YYYY-MM-DD",
     "phoneNumberMobile":"12345678", "nationalIdentityNumber":"12345678901",
     "userType":"STANDARD", "department":{{"id":DEPT_ID}}}}
4. POST /employee/employment {{"employee":{{"id":EMP_ID}}, "startDate":"YYYY-MM-DD"}}
5. If salary mentioned: POST /salary/transaction or note it

CRITICAL: nationalIdentityNumber is a valid field on Employee. Include it if found in PDF.
CRITICAL: Do NOT include employmentType or percentageOfFullTimeEquivalent on /employee/employment — only employee.id and startDate.
CRITICAL: Phone field is phoneNumberMobile on Employee (NOT phoneNumber).
""",

    "bank_reconciliation_csv": """\
## Bank Reconciliation from CSV (bankavstemminger)
1. Parse the CSV attachment — each row is a transaction (date, description, amount, reference)
2. GET /invoice?invoiceDateFrom=X&invoiceDateTo=Y&fields=id,invoiceNumber,amount,amountOutstanding,customer
   - Match incoming payments (positive amounts) to customer invoices
3. GET /supplierInvoice?fields=id,invoiceNumber,amount — match outgoing payments (if any exist)
4. For each matched customer payment:
   - GET /invoice/paymentType?fields=id,description — get payment type ID first
   - PUT /invoice/ID/:payment?paymentDate=DATE&paymentTypeId=X&paidAmount=AMOUNT
5. Handle partial payments: paidAmount can be less than invoice total
6. Unmatched transactions: create vouchers via POST /ledger/voucher for unknown items
   - Debit/credit appropriate accounts (1920 bank, expense/revenue accounts)

IMPORTANT: GET /invoice/paymentType first for paymentTypeId.
IMPORTANT: Both invoiceDateFrom AND invoiceDateTo are required on GET /invoice.
""",

    "ledger_analysis": """\
## Ledger Analysis → Create Projects
1. GET /ledger/account?fields=id,number,name — list all accounts
2. For each expense account (5xxx-8xxx), GET posting totals:
   GET /ledger/posting?accountId=X&dateFrom=2026-01-01&dateTo=2026-01-31 (January)
   GET /ledger/posting?accountId=X&dateFrom=2026-02-01&dateTo=2026-02-28 (February)
3. Calculate increase: feb_total - jan_total for each account
4. Sort by increase, pick top 3
5. For each top account:
   - GET /department?fields=id,name&count=1
   - POST /employee (if no project manager exists) with department
   - PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=ID&template=ALL_PRIVILEGES
   - POST /project {{"name":"account_name", "startDate":"2026-01-01", "isInternal":true, "projectManager":{{"id":EMP_ID}}}}
   - POST /project/projectActivity for each project if needed

IMPORTANT: Query existing ledger data BEFORE creating new entities.
""",

    "supplier_invoice_pdf": """\
## Supplier Invoice from PDF
1. Read the PDF — extract: supplier name, org number, invoice number, date, due date, amount, expense account
2. POST /supplier {{"name":"X", "organizationNumber":"Y"}} (create if not exists)
3. GET /ledger/account?number=EXPENSE_ACCT&fields=id (e.g. 6500, 6700, 7100)
4. GET /ledger/account?number=2400&fields=id (leverandørgjeld/accounts payable)
5. POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Leverandørfaktura INV-XXX fra SupplierName",
     "postings":[
       {{"row":1, "account":{{"id":EXPENSE_ACCT_ID}}, "amountGross":FULL_AMOUNT, "amountGrossCurrency":FULL_AMOUNT, "vatType":{{"id":1}}, "supplier":{{"id":SUPPLIER_ID}}}},
       {{"row":2, "account":{{"id":AP_ACCT_ID}}, "amountGross":-FULL_AMOUNT, "amountGrossCurrency":-FULL_AMOUNT, "vatType":{{"id":0}}, "supplier":{{"id":SUPPLIER_ID}}}}
     ]}}

AMOUNT RULES: amountGross = the FULL/GROSS amount (including VAT). Both postings use the SAME absolute amount. Tripletex calculates VAT split automatically based on vatType.
CRITICAL: NEVER use POST /supplierInvoice (always 500). Go directly to POST /ledger/voucher.
""",
    "project_lifecycle": """\
## Complete Project Lifecycle
For tasks like "Execute the complete project lifecycle":

1. POST /customer (create customer)
2. GET /department (for employees)
3. POST /employee × N (create project team members with department)
4. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=X&template=ALL_PRIVILEGES (for project manager)
5. POST /project {{"name":"X", "startDate":"YYYY-MM-DD", "projectManager":{{"id":PM_ID}}, "customer":{{"id":CUST_ID}}}}
6. POST /timesheet/entry × N (register hours for each employee: employee.id, project.id, activity.id, date, hours)
   - GET /activity?isProjectActivity=true first
   - hours field is the number of hours
   - comment field for description (NOT description)
7. Supplier cost: POST /supplier → GET /ledger/account → POST /ledger/voucher (with project ref if needed)
8. Customer invoice: POST /order {{"customer":{{"id":X}}, "orderDate":"Y", "deliveryDate":"Y", "orderLines":[{{"description":"Project work", "count":1, "unitPriceExcludingVatCurrency": CALCULATED_AMOUNT}}]}}
9. PUT /order/ID/:invoice?invoiceDate=Y&invoiceDueDate=Y&sendToCustomer=false

CRITICAL: Invoice amount must be MANUALLY CALCULATED:
- If hourly rate task: amount = total_hours × hourly_rate
- If budget task: amount = budget_amount or percentage of budget
- Tripletex does NOT auto-calculate from timesheet hours
""",
    "employment_details": """\
## Employment Details (salary, working hours, position)
For tasks like "Configure employment with salary and working hours":

CRITICAL: Employment URL is GET /employee/employment?employeeId=ID
NOT GET /employee/ID/employment (returns 404!)

Employment details are on a SEPARATE endpoint from basic employment:
POST /employee/employment/details {{"employment":{{"id":EMPLOYMENT_ID}}, "date":"YYYY-MM-DD", "annualSalary":AMOUNT, "percentageOfFullTimeEquivalent":1.0}}

Available fields on employment/details:
- annualSalary (float) — yearly salary
- hourlyWage (float) — hourly rate
- percentageOfFullTimeEquivalent (float) — FTE, e.g. 1.0 = 100%
- employmentType (int) — GET /employee/employment/employmentType for valid IDs
- workingHoursScheme (int) — GET /employee/employment/workingHoursScheme for valid IDs
- remunerationType (int) — GET /employee/employment/remunerationType for valid IDs
- occupationCode (int) — GET /employee/employment/occupationCode for valid IDs
- date (string) — effective date

Flow for full employee setup from offer letter:
1. POST /employee (firstName, lastName, email, dateOfBirth, phoneNumberMobile, userType, department)
2. POST /employee/employment {{"employee":{{"id":X}}, "startDate":"YYYY-MM-DD"}}
3. GET /employee/employment/employmentType (find valid type)
4. GET /employee/employment/workingHoursScheme (find valid scheme)
5. POST /employee/employment/details {{"employment":{{"id":EMPL_ID}}, "date":"YYYY-MM-DD", "annualSalary":X, "percentageOfFullTimeEquivalent":1.0, "employmentType":TYPE_ID, "workingHoursScheme":SCHEME_ID}}

NOTE: employment (step 2) and employment/details (step 5) are DIFFERENT endpoints!

FORBIDDEN fields on /employee/employment/details:
- position (does NOT exist)
- title, jobTitle (do NOT exist)
- role (does NOT exist — that's on Employee, not EmploymentDetails)
- userType (does NOT exist on details)

employmentType MUST be an integer ID (NOT a string, NOT an object).
GET /employee/employment/employmentType first to find valid IDs.
Example: {{"employment":{{"id":X}}, "date":"YYYY-MM-DD", "annualSalary":500000, "employmentType":1, "percentageOfFullTimeEquivalent":100.0}}
""",

    "year_end_closing": """\
## Year-End Closing (årsavslutning / encerramento anual)
Complex multi-step task. Each step creates a separate voucher.

### Step 1: Depreciation (avskrivning / depreciação)
For each asset, calculate: annual_depreciation = acquisition_cost / useful_life_years
Then create a voucher:
- Debit: depreciation EXPENSE account (e.g. 6010) with the calculated amount
- Credit: accumulated depreciation account (e.g. 1209) with negative amount
Example: Asset 375600 NOK, 8 years → 375600/8 = 46950 NOK per year
POST /ledger/voucher {"date":"2025-12-31", "description":"Avskrivning Kontormaskiner", "postings":[
  {"row":1, "account":{"id":EXPENSE_6010_ID}, "amountGross":46950, "amountGrossCurrency":46950, "vatType":{"id":0}},
  {"row":2, "account":{"id":ACCUM_1209_ID}, "amountGross":-46950, "amountGrossCurrency":-46950, "vatType":{"id":0}}
]}
CRITICAL: Create a SEPARATE voucher for EACH asset, not one combined.

### Step 2: Reverse prepaid expenses (reversere forhåndsbetalte utgifter)
Move from prepaid (1700) to expense account:
- Debit: relevant expense account (e.g. 6xxx-7xxx)
- Credit: prepaid account (1700) with negative amount
POST /ledger/voucher with description "Reversering forhåndsbetalte utgifter"

### Step 3: Tax provision (skatteavsetning / provisão fiscal)
Calculate: tax = taxable_income * 0.22 (Norwegian corporate tax rate 22%)
Then create voucher:
- Debit: tax expense account (8700) with tax amount
- Credit: tax payable account (2920) with negative tax amount

### Norwegian Chart of Accounts Reference
- 1200-1299: Fixed assets (anleggsmidler)
- 1209: Accumulated depreciation (akkumulerte avskrivninger)
- 1700: Prepaid expenses (forhåndsbetalte kostnader)
- 2920: Tax payable (betalbar skatt)
- 6010: Depreciation expense (avskrivning)
- 8700: Tax expense (skattekostnad)

### Key Rules
- ALL amounts must be numbers (not strings!)
- amountGross = the full amount (vatType 0 for balance sheet accounts)
- Each depreciation = separate voucher with 2 postings
- Postings MUST sum to zero
- Date should be year-end: YYYY-12-31
- GET /ledger/account?number=XXXX&fields=id for each account first
""",
}

# Aliases for common misspellings / alternative names
API_GUIDES["travel"] = API_GUIDES["travel_expense"]
API_GUIDES["expense"] = API_GUIDES["travel_expense"]
API_GUIDES["reiseregning"] = API_GUIDES["travel_expense"]
API_GUIDES["faktura"] = API_GUIDES["invoice"]
API_GUIDES["bilag"] = API_GUIDES["voucher"]
API_GUIDES["kunde"] = API_GUIDES["customer"]
API_GUIDES["ansatt"] = API_GUIDES["employee"]
API_GUIDES["leverandor"] = API_GUIDES["supplier"]
API_GUIDES["leverandør"] = API_GUIDES["supplier"]
API_GUIDES["prosjekt"] = API_GUIDES["project"]
API_GUIDES["avdeling"] = API_GUIDES["department"]
API_GUIDES["produkt"] = API_GUIDES["product"]
API_GUIDES["kontakt"] = API_GUIDES["contact"]
API_GUIDES["innbetaling"] = API_GUIDES["payment"]
API_GUIDES["kreditnota"] = API_GUIDES["credit_note"]
API_GUIDES["purring"] = API_GUIDES["reminder"]
API_GUIDES["timeføring"] = API_GUIDES["timesheet"]
API_GUIDES["lønn"] = API_GUIDES["salary"]
API_GUIDES["ansettelse"] = API_GUIDES["employment"]
API_GUIDES["åpningsbalanse"] = API_GUIDES["opening_balance"]
API_GUIDES["leverandørfaktura"] = API_GUIDES["supplier_invoice"]
API_GUIDES["innkjøpsordre"] = API_GUIDES["purchase_order"]
API_GUIDES["anleggsmiddel"] = API_GUIDES["asset"]
API_GUIDES["bankavstemming"] = API_GUIDES["bank_reconciliation"]
API_GUIDES["dimensjon"] = API_GUIDES["dimensions"]
API_GUIDES["fastpris"] = API_GUIDES["fixed_price_project"]
API_GUIDES["order"] = API_GUIDES["invoice"]  # order creation is part of invoice flow
API_GUIDES["bank"] = "POST /bank {\"accountNumber\":\"86011117947\", \"name\":\"Driftskonto\"}\nUsed to register a bank account when invoicing fails with 'bankkontonummer' error."
API_GUIDES["account"] = API_GUIDES["voucher"]  # account lookups covered in voucher guide
API_GUIDES["konto"] = API_GUIDES["voucher"]

# Tier 3 aliases
API_GUIDES["receipt"] = API_GUIDES["receipt_voucher"]
API_GUIDES["kvittering"] = API_GUIDES["receipt_voucher"]
API_GUIDES["contract"] = API_GUIDES["employment_contract_pdf"]
API_GUIDES["arbeidskontrakt"] = API_GUIDES["employment_contract_pdf"]
API_GUIDES["reconciliation"] = API_GUIDES["bank_reconciliation_csv"]
API_GUIDES["avstemming"] = API_GUIDES["bank_reconciliation_csv"]
API_GUIDES["bankavstemminger"] = API_GUIDES["bank_reconciliation_csv"]
API_GUIDES["regnskapsanalyse"] = API_GUIDES["ledger_analysis"]
API_GUIDES["leverandørfaktura_pdf"] = API_GUIDES["supplier_invoice_pdf"]
API_GUIDES["lifecycle"] = API_GUIDES["project_lifecycle"]
API_GUIDES["salary_setup"] = API_GUIDES["employment_details"]
API_GUIDES["arbeidskontrakt_detaljer"] = API_GUIDES["employment_details"]
API_GUIDES["arsavslutning"] = API_GUIDES["year_end_closing"]
API_GUIDES["encerramento"] = API_GUIDES["year_end_closing"]
API_GUIDES["jahresabschluss"] = API_GUIDES["year_end_closing"]
API_GUIDES["cierre"] = API_GUIDES["year_end_closing"]
API_GUIDES["depreciation"] = API_GUIDES["year_end_closing"]
API_GUIDES["avskrivning"] = API_GUIDES["year_end_closing"]

API_GUIDES["currency_exchange"] = """\
## Currency Exchange / Agio (valutadifferanse)
Task: Sent invoice in EUR at rate X, customer paid at rate Y, book the exchange difference (agio).

Steps:
1. POST /customer (create customer with org number)
2. GET /currency?code=EUR&fields=id,code,factor — get currency ID (fields: id, code, description, displayName, factor — NOT name!)
   EUR id is typically 5, NOK id is 1.
3. POST /order with currency.id, orderLines with amount in foreign currency
4. PUT /order/ORDER_ID/:invoice with invoiceDate, invoiceDueDate
5. GET /invoice/paymentType for payment type ID
6. PUT /invoice/INV_ID/:payment with paymentDate, paidAmount (in NOK = foreign amount × new rate)
7. Book agio: POST /ledger/voucher
   - If gain (new rate > old rate): debit 1500 (kundefordring), credit 8060 (annen finansinntekt)
   - If loss (new rate < old rate): debit 8160 (annen finanskostnad), credit 1500

Example agio voucher (loss of 500 NOK):
POST /ledger/voucher {"date":"YYYY-MM-DD", "description":"Agiotap valutadifferanse",
  "postings":[
    {"row":1, "account":{"id":8160_ACCT_ID}, "amountGross":500, "amountGrossCurrency":500, "vatType":{"id":0}},
    {"row":2, "account":{"id":1500_ACCT_ID}, "amountGross":-500, "amountGrossCurrency":-500, "vatType":{"id":0}}
  ]}
"""
API_GUIDES["agio"] = API_GUIDES["currency_exchange"]
API_GUIDES["valutakurs"] = API_GUIDES["currency_exchange"]
API_GUIDES["exchange_rate"] = API_GUIDES["currency_exchange"]

API_GUIDES["overdue_invoice_reminder"] = """\
## Overdue Invoice + Reminder Fee + Partial Payment
Competition prompt: "Find the overdue invoice, post reminder fee of 35 NOK (debit 1500, credit 3400), create invoice for fee, send it, register partial payment"

Flow:
1. GET /invoice?invoiceDateFrom=2026-01-01&invoiceDateTo=2026-12-31&fields=id,invoiceNumber,amount,amountOutstanding,invoiceDueDate,customer
   - Find invoice where amountOutstanding > 0 AND invoiceDueDate < today
2. POST /ledger/voucher — reminder fee posting
   - Debit 1500 (Kundefordringer): amountGross = 35
   - Credit 3400 (use whatever account the prompt says): amountGross = -35
   - GET /ledger/account?number=1500 and ?number=3400 first for IDs
3. POST /order + PUT /:invoice — create invoice for the fee
4. PUT /invoice/:send?sendType=EMAIL
5. PUT /invoice/{overdue_id}/:payment — partial payment
   - GET /invoice/paymentType first
   - paidAmount = 5000 (or whatever prompt says)
"""
API_GUIDES["overdue"] = API_GUIDES["overdue_invoice_reminder"]
API_GUIDES["forfalt"] = API_GUIDES["overdue_invoice_reminder"]
API_GUIDES["impaye"] = API_GUIDES["overdue_invoice_reminder"]

API_GUIDES["ledger_correction"] = """\
## Ledger Correction (feilretting i regnskap)
Steps for correcting voucher errors:

1. GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD to find vouchers
   - MUST include dateFrom AND dateTo (both required!)
2. For wrong account: POST new correcting voucher (reverse original + post correct)
   - Create voucher with reversed postings of the original (swap debit/credit signs)
   - Then create another voucher with the correct account numbers
3. For duplicate: PUT /ledger/voucher/ID/:reverse?date=YYYY-MM-DD
   - date is a QUERY param (not body!)
4. For missing VAT: POST new voucher with VAT posting
   - Debit the VAT receivable account (e.g. 2710), credit the expense account
5. For wrong amount: POST correcting voucher for the difference
   - Only book the delta between correct and incorrect amount

CRITICAL: Always GET /ledger/account?number=XXXX&fields=id before creating vouchers.
CRITICAL: Postings MUST sum to zero. Use vatType.id=0 for balance sheet corrections.
"""
API_GUIDES["feilretting"] = API_GUIDES["ledger_correction"]

# Fields to preserve in _compact_response
_ESSENTIAL_FIELDS = frozenset({
    "id", "version", "name", "number", "firstName", "lastName",
    "email", "invoiceNumber", "amount", "amountOutstanding",
    "status", "orderId", "organizationNumber", "bankAccountNumber",
    "startDate", "endDate", "dateOfBirth", "phoneNumber",
    "phoneNumberMobile", "isCustomer", "isSupplier", "isInternal",
    "isFixedPrice", "fixedprice", "description", "userType",
    "department", "projectManager", "customer", "supplier",
    "employee", "invoiceDate", "invoiceDueDate", "postalAddress",
    "priceExcludingVatCurrency", "employeeNumber", "departmentNumber",
    # Additional scoring-relevant fields
    "physicalAddress", "deliveryAddress", "count", "unitPriceExcludingVatCurrency",
    "orderDate", "deliveryDate", "orderLines", "comment", "hours",
    "title", "travelDetails", "isDayTrip", "isForeignTravel",
    "departureDate", "returnDate", "departureFrom", "destination",
    "purpose", "costCategory", "amountCurrencyIncVat", "paymentType",
    "vatType", "row", "account", "amountGross", "amountGrossCurrency",
    "postings", "voucher", "type", "displayName", "dimensionName",
    "dimensionIndex", "freeAccountingDimension1", "acquisitionCost",
    "dateOfAcquisition", "year", "month", "payslips",
})

_LIST_ESSENTIAL_FIELDS = frozenset({
    "id", "version", "name", "number", "type", "description",
    "bankAccountNumber", "firstName", "lastName", "email",
    "startDate", "status", "isProjectActivity",
    "organizationNumber", "phoneNumber", "phoneNumberMobile",
    "departmentNumber", "accountNumber", "paymentTypeId",
})


# ── Agent execution ──────────────────────────────────────────────────

async def tool_agent_solve(
    prompt: str,
    files: list[dict] | None,
    client: TripletexClient,
    deadline: float,
) -> bool:
    """Run the tool-use agent. Returns True if task completed without errors."""

    # Use Flash for speed — Pro is too slow (15-20s/turn = timeout on complex tasks)
    # Flash: 2-5s/turn, handles API routing fine, gives us 5x more turns in same budget
    model_name = "gemini-2.5-flash"
    location = "europe-north1"
    vertexai.init(project="ainm26osl-710", location=location)
    model = GenerativeModel(model_name, system_instruction=SYSTEM_PROMPT, tools=TOOLS)
    logger.info(f"Tool agent using {model_name} ({location})")

    # Build initial user message
    parts = []
    if files:
        for f in files:
            try:
                data = base64.b64decode(f.get("content_base64", ""))
                parts.append(Part.from_data(data=data, mime_type=f.get("mime_type", "application/octet-stream")))
                parts.append(Part.from_text(f"[Attached: {f.get('filename', 'file')}]"))
            except Exception:
                pass
    parts.append(Part.from_text(f"Execute this accounting task:\n\n{prompt}"))

    chat = model.start_chat()
    had_errors = False
    for turn in range(MAX_TURNS):
        remaining = deadline - time.monotonic()
        if remaining < DEADLINE_BUFFER:
            logger.warning(f"Tool agent: deadline approaching ({remaining:.0f}s), stopping at turn {turn}")
            break
        # Hard limit on WRITE calls only (GET is free per scoring rules)
        write_count = sum(1 for c in getattr(client, 'call_log', []) if c.get('method') in ('POST', 'PUT', 'DELETE'))
        if write_count > 20:
            logger.warning(f"Tool agent: hard limit — {write_count} write calls, stopping at turn {turn}")
            break

        # Send message to LLM
        try:
            response = await asyncio.wait_for(
                chat.send_message_async(
                    parts,
                    generation_config={"temperature": 0.0, "max_output_tokens": 4096},
                ),
                timeout=max(5.0, min(60.0, remaining - DEADLINE_BUFFER)),
            )
        except asyncio.TimeoutError:
            logger.error(f"Tool agent: LLM timeout at turn {turn} ({model_name})")
            had_errors = True
            break
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "Resource exhausted" in error_str:
                logger.warning(f"Tool agent: 429 rate limit at turn {turn}, retrying in 3s")
                await asyncio.sleep(3)
                try:
                    response = await asyncio.wait_for(
                        chat.send_message_async(parts, generation_config={"temperature": 0.0, "max_output_tokens": 4096}),
                        timeout=max(5.0, min(60.0, remaining - DEADLINE_BUFFER)),
                    )
                except Exception as e2:
                    logger.error(f"Tool agent: retry also failed: {e2}")
                    had_errors = True
                    break
            else:
                logger.error(f"Tool agent: LLM error at turn {turn}: {e}")
                had_errors = True
                break

        # Check if LLM wants to call functions
        candidates = response.candidates
        if not candidates:
            logger.info("Tool agent: no candidates, assuming done")
            break

        content = candidates[0].content
        has_function_calls = any(
            hasattr(part, 'function_call') and part.function_call is not None and part.function_call.name
            for part in content.parts
        )

        if not has_function_calls:
            text = response.text if hasattr(response, 'text') else ""
            logger.info(f"Tool agent: text response at turn {turn} (done): {text[:200]}")
            break

        # Execute all function calls (local ones sync, API calls batched for parallel)
        function_responses = []
        api_calls_to_execute = []
        for part in content.parts:
            if not hasattr(part, 'function_call') or part.function_call is None or not part.function_call.name:
                continue

            fc = part.function_call
            fn_name = fc.name
            args = dict(fc.args) if fc.args else {}

            # Handle get_api_guide locally (no HTTP request)
            if fn_name == "get_api_guide":
                topic = str(args.get("topic", "")).strip().lower()
                guide = API_GUIDES.get(topic)
                if guide:
                    logger.info(f"Tool agent turn {turn}: get_api_guide({topic}) -> found")
                    result_text = guide
                else:
                    available = sorted(set(k for k, v in API_GUIDES.items() if k == v or k not in (
                        "travel", "expense", "reiseregning", "faktura", "bilag",
                        "kunde", "ansatt", "leverandor", "leverandør", "prosjekt",
                        "avdeling", "produkt", "kontakt", "innbetaling", "kreditnota",
                        "purring", "timeføring", "lønn", "ansettelse", "åpningsbalanse",
                        "leverandørfaktura", "innkjøpsordre", "anleggsmiddel",
                        "bankavstemming", "dimensjon", "fastpris"
                    )))
                    result_text = f"Topic '{topic}' not found. Available topics: {', '.join(available)}"
                    logger.info(f"Tool agent turn {turn}: get_api_guide({topic}) -> not found")
                function_responses.append(
                    Part.from_function_response(
                        name=fn_name,
                        response={"result": result_text},
                    )
                )
                continue

            # Handle get_api_schema locally (no HTTP request)
            if fn_name == "get_api_schema":
                entity = str(args.get("entity", "")).strip()
                schema = _FIELD_REF.get(entity)
                if schema:
                    result_text = f"Fields for {entity}: {json.dumps(schema)}"
                    logger.info(f"Tool agent turn {turn}: get_api_schema({entity}) -> {len(schema)} fields")
                else:
                    available = sorted(_FIELD_REF.keys())
                    result_text = f"Entity '{entity}' not found. Available: {', '.join(available)}"
                    logger.info(f"Tool agent turn {turn}: get_api_schema({entity}) -> not found")
                function_responses.append(
                    Part.from_function_response(
                        name=fn_name,
                        response={"result": result_text},
                    )
                )
                continue

            # Collect HTTP API calls for parallel execution
            path = args.get("path", "")
            body = args.get("body")
            params = args.get("params")
            if path and not path.startswith("/"):
                path = "/" + path
            method_map = {
                "tripletex_get": "GET",
                "tripletex_post": "POST",
                "tripletex_put": "PUT",
                "tripletex_delete": "DELETE",
            }
            method = method_map.get(fn_name, "GET")
            api_calls_to_execute.append((fn_name, method, path, body, params))

        # Execute ALL API calls in parallel (huge speed win when model emits 2-5 calls per turn)
        if api_calls_to_execute:
            async def _exec_one(fn_name, method, path, body, params):
                logger.info(f"Tool agent turn {turn}: {method} {path}")
                try:
                    return fn_name, await client.request(method, path, body=body, params=params)
                except Exception as e:
                    return fn_name, {"ok": False, "status_code": 0, "data": {"error": str(e)}}

            results = await asyncio.gather(
                *[_exec_one(fn, m, p, b, pa) for fn, m, p, b, pa in api_calls_to_execute]
            )

            for fn_name, result in results:
                ok = result.get("ok", False)
                status = result.get("status_code", 0)
                data = result.get("data", {})
                if not ok:
                    # Only count write failures as errors — GET failures are just exploration
                    if fn_name != "tripletex_get":
                        had_errors = True
                    logger.warning(f"Tool agent: {fn_name} -> {status} FAIL")
                    record_error(fn_name, data, prompt)
                summary = _compact_response(data, ok)
                function_responses.append(
                    Part.from_function_response(
                        name=fn_name,
                        response={"result": summary},
                    )
                )

        # Send function results back to LLM
        # Preserve non-function-call parts (text, thought signatures) from
        # the model response — Gemini 3.1 Pro uses encrypted thought signatures
        # that must be echoed back to maintain chain-of-thought coherence.
        preserved_parts = [
            p for p in content.parts
            if not (hasattr(p, 'function_call') and p.function_call is not None and p.function_call.name)
        ]
        parts = preserved_parts + function_responses

    # Determine success: check if we made at least one successful write call
    # and the LAST write call succeeded (recovery from earlier errors is OK)
    call_log = getattr(client, 'call_log', [])
    write_calls = [c for c in call_log if c.get('method') in ('POST', 'PUT', 'DELETE')]
    has_successful_write = any(c.get('ok') for c in write_calls)
    last_write_ok = write_calls[-1].get('ok', False) if write_calls else False
    success = has_successful_write and last_write_ok

    if success and not had_errors:
        compile_template(prompt, call_log)

    return success


def _compact_response(data: dict, ok: bool, max_len: int = 1500) -> dict:
    """Compact API response to save tokens while keeping essential info."""
    if not ok:
        return {"ok": False, "error": json.dumps(data, ensure_ascii=False, default=str)[:max_len]}

    if isinstance(data, dict):
        if "value" in data:
            val = data["value"]
            if isinstance(val, dict):
                compact = {k: v for k, v in val.items() if k in _ESSENTIAL_FIELDS}
                return {"ok": True, "value": compact}
            return {"ok": True, "value": val}

        if "values" in data:
            vals = data["values"]
            if isinstance(vals, list):
                compact_list = []
                for item in vals[:10]:
                    if isinstance(item, dict):
                        compact_list.append({k: v for k, v in item.items() if k in _LIST_ESSENTIAL_FIELDS})
                    else:
                        compact_list.append(item)
                truncation_note = f"(showing {min(10, len(vals))}/{len(vals)} results)" if len(vals) > 10 else ""
                result = {"ok": True, "count": len(vals), "values": compact_list}
                if truncation_note:
                    result["note"] = truncation_note
                return result

    return {"ok": True, "data": json.dumps(data, ensure_ascii=False, default=str)[:max_len]}

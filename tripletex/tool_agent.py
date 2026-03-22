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
    description="Get detailed API documentation for a specific topic. Call this BEFORE making API calls you're unsure about. IMPORTANT: For complex tasks, use the SPECIFIC guide (e.g. 'currency_exchange' for EUR invoices, 'month_end_closing' for period closing, 'salary' for payroll). Topics: customer, employee, invoice, voucher, travel_expense, project, supplier, product, department, contact, payment, credit_note, reminder, send_invoice, timesheet, salary, employment, opening_balance, supplier_invoice, purchase_order, asset, bank_reconciliation, dimensions, fixed_price_project, update_entity, receipt_voucher, employment_contract_pdf, bank_reconciliation_csv, ledger_analysis, supplier_invoice_pdf, currency_exchange, currency_payment, ledger_correction, overdue_invoice_reminder, month_end_closing, project_lifecycle, year_end_closing",
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
2. Call get_api_guide for the MOST SPECIFIC topic. Use these guide names:
   - Currency/EUR/exchange rate tasks → get_api_guide("currency_exchange") AND get_api_guide("currency_payment")
   - Month-end/accrual/periodisering/salary accrual → get_api_guide("month_end_closing")
   - Overdue invoice + reminder fee → get_api_guide("overdue_invoice_reminder")
   - Bank reconciliation CSV → get_api_guide("bank_reconciliation_csv")
   - Salary/payroll/lønn → get_api_guide("salary")
   - Year-end closing → get_api_guide("year_end_closing")
   - Voucher correction → get_api_guide("voucher_correction")
   - Salary/payroll → get_api_guide("salary")
   - Employee from PDF → get_api_guide("employee")
   - Supplier invoice from PDF → get_api_guide("supplier_invoice")
   - Month-end closing → get_api_guide("month_end_closing")
   - Project lifecycle → get_api_guide("project_lifecycle")
   - Ledger analysis → get_api_guide("ledger_analysis")
3. You may call get_api_guide and get_api_schema as needed. Prefer calling ONCE with the most specific topic.
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
- PUT /company/modules (returns 405)
- PUT /salary/payslip/ID (returns 405 — payslips are READ-ONLY after creation)
- PUT /salary/transaction/ID (returns 405 — transactions are READ-ONLY)
- POST /salary/transaction/line (returns 405)
- GET /salary/payslip/ID/line (returns 404)
- PUT /invoice/ID/:reverse (does NOT exist — use PUT /ledger/voucher/ID/:reverse instead)
- GET /invoice with field "totalAmountExcludingVatCurrency" or "description" (invalid fields)
- DELETE /employee/employment/ID (returns 405 — employments cannot be deleted)
- PUT /employee/employment/ID with "department" field (doesn't exist on employment — department is on employee)
- Some accounts are LOCKED to vatType 0: 1500, 1920, 2400, 3400, 7350, 8060, 8160 and other non-VAT accounts. If you get "Kontoen er låst til mva-kode 0", use vatType:{{"id":0}} for that posting
- ALWAYS use GET /ledger/account?number=X&fields=id,vatType to check if account has locked vatType BEFORE posting
- PUT /invoice/ID/:send MUST include sendType param (e.g. sendType=EMAIL)
- GET /currency: fields are id, code, description, displayName, factor (NOT name — causes 400)
- Employment lookup: GET /employee/employment?employeeId=ID (NOT /employee/ID/employment)

MANDATORY FIELD RULES (violating these = instant 422):
- Product: field is "number" (NOT productNumber, NOT productNo)
- Product vatType: must be {{"id": N}} where N = 3 (25%), 33 (15% food), 31 (12%), 5 (0%)
- Voucher: "description" is REQUIRED (not optional)
- Employee: phone is "phoneNumberMobile" (NOT phone, NOT phoneNumber, NOT mobileNumber)
- POST /supplierInvoice: valid fields are invoiceNumber, invoiceDate, invoiceDueDate, supplier.id, voucher (with postings), amountCurrency. Do NOT send orderDate, deliveryDate, dueDate, amount, orderLines
- POST /employee: MUST include userType:"STANDARD" AND department:{{"id":X}} (GET /department first!)
- POST /travelExpense: isDayTrip and isForeignTravel go INSIDE travelDetails (NOT top-level body)
- POST /travelExpense/cost: amountCurrencyIncVat is REQUIRED. costCategory must be {{"id":X}} object (NOT string)
- GET /invoice: MUST include invoiceDateFrom AND invoiceDateTo params (both required)
- PUT /:invoice: MUST include invoiceDueDate param (invoiceDate + 14 days if not specified)

ACCOUNT NUMBER → VATTYPE RULES (Norwegian chart of accounts):
- 1000-1999 (balance sheet): vatType {{"id": 0}} ALWAYS
- 2000-2999 (liabilities): vatType {{"id": 0}} ALWAYS
- 3000-3999 (revenue): vatType {{"id": 3}} (outgoing 25%) — except 3400 which is locked to 0
- 4000-4999 (cost of goods): vatType {{"id": 1}} (incoming 25%)
- 5000-5999 (payroll): vatType {{"id": 0}} ALWAYS
- 6000-6999 (operating expenses): vatType {{"id": 1}} (incoming 25%)
- 7000-7999 (other opex): vatType {{"id": 0}} ALWAYS
- 8000-8999 (finance): vatType {{"id": 0}} ALWAYS
CRITICAL: Using wrong vatType on locked accounts causes "Kontoen er låst til mva-kode 0" error!

POSTING ENTITY REQUIREMENTS:
- Account 1500 (Kundefordringer) postings MUST include customer: {{"id": X}}
- Account 2400 (Leverandørgjeld) postings MUST include supplier: {{"id": X}}
- Account 5000/5020/2910 (payroll) postings MUST include employee: {{"id": X}}
- PUT /:reverse: date goes as QUERY param (not body)
- GET /ledger/voucher: MUST include dateFrom AND dateTo params (both required). dateTo must be AFTER dateFrom (not same day! use dateFrom=2026-03-20&dateTo=2026-03-21)
- Voucher postings: row starts from 1, MUST include amountGrossCurrency AND vatType
- ProjectHourlyRate: rate field is "fixedRate" (NOT hourlyRate). hourlyRateModel is a string like "TYPE_FIXED_HOURLY_RATE"
- orderLine.vatType MUST be an object {{"id": N}}, NOT a bare number. Common IDs: 3=25% outgoing, 33=15% food, 5=0% exempt
- NEVER PUT /activity — activities are read-only. Use GET /activity to find existing ones, don't try to modify them.
- POST /activity requires activityType field. But NEVER create activities — use GET /activity?isProjectActivity=true to find existing ones.
- /activityType endpoint does NOT exist (404). Activity types are predefined.
- If timesheet date < project startDate, PUT /project to change startDate (NOT PUT /activity)

ADDITIONAL ENDPOINTS YOU MAY NEED:
- POST /travelExpense/mileageAllowance — for km-godtgjørelse/mileage. Fields: travelExpense, rateTypeId, km, rate, date
- POST /travelExpense/perDiemCompensation — for diett/per diem. Alternative to manual cost entry.
- POST /supplierInvoice/ID/:addPayment — pay a supplier invoice. Params: paymentDate, paymentTypeId, paidAmount
- GET /ledger/paymentTypeOut?fields=id,description — outgoing payment types (for supplier payments)
- PUT /travelExpense/:createVouchers?id=X — book travel expense to ledger
- POST /employee/nextOfKin — emergency contacts: employee, firstName, lastName, phoneNumber
- POST /employee/hourlyCostAndRate — set hourly rates: employee, date, rate, costRate
- GET /currency/ID/exchangeRate?date=YYYY-MM-DD — look up exchange rate for a date
- GET /balanceSheet?dateFrom=X&dateTo=Y — balance sheet query
- POST /employee/employment/leaveOfAbsence — sick leave/permisjon: employment, type, startDate, percentage
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
- dateOfBirth: include if mentioned, format YYYY-MM-DD. REQUIRED for salary/employment.
- employeeNumber: writable — include if task specifies an employee number
- address: {{"addressLine1":"Street 1", "postalCode":"1234", "city":"Oslo"}} — include if task mentions employee address
- nationalIdentityNumber: 11-digit Norwegian personal ID — include if task mentions personnummer/fødselsnummer
- bankAccountNumber: employee bank account — include if mentioned (needed for salary)
- comments: free text — include if task has notes about the employee
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
- Valid fields: employment, date, employmentType, employmentForm, remunerationType, workingHoursScheme, shiftDurationHours, occupationCode, percentageOfFullTimeEquivalent, annualSalary, hourlyWage
- monthlySalary is READ-ONLY (auto-calculated from annualSalary/12) — do NOT set it
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
- INVALID fields (cause 400): voucherNumber, amountExVat, totalAmount, totalAmountCurrency, exchangeRate, status, description

To reverse/undo a payment:
1. Create the full invoice flow first (customer → order → invoice → payment)
2. GET /ledger/voucher?dateFrom=YYYY-MM-DD&dateTo=YYYY-MM-DD&fields=id,date,description — find the payment voucher
3. PUT /ledger/voucher/VOUCHER_ID/:reverse?date=YYYY-MM-DD — reverse it (date is QUERY param!)
""",

    "currency_payment": """\
## Currency Payment / Disagio (valutaoppgjør)
IMPORTANT: The sandbox starts EMPTY. There is NO pre-existing invoice or customer. You MUST create everything from scratch!

Steps:
1. POST /customer (create customer with name, orgNo, isCustomer:true)
2. POST /order (create order with orderLines — unitPriceExcludingVatCurrency = foreign_amount × invoice_exchange_rate)
3. PUT /order/ID/:invoice (invoice the order)
4. GET /invoice/paymentType (get payment type ID)
5. PUT /invoice/INV_ID/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=AMOUNT_NOK&paidAmountCurrency=AMOUNT_FOREIGN_CURRENCY
   - paidAmount = amount in NOK (foreign amount × payment exchange rate)
   - paidAmountCurrency = amount in foreign currency (e.g. EUR)
   - BOTH paidAmount AND paidAmountCurrency are REQUIRED for foreign currency payments
3. Post disagio voucher for the exchange rate difference:
   - Disagio (loss): invoice rate > payment rate → company loses money
     - Debit 8160 (Agio/disagio) with the NOK difference
     - Credit 1500 (Customer receivables) with negative NOK difference
   - Agio (gain): invoice rate < payment rate → company gains money
     - Debit 1500 (Customer receivables) with the NOK difference
     - Credit 8060 (Agio income) with negative NOK difference
   - Calculation: difference = (invoice_rate - payment_rate) × foreign_amount
   - Both accounts (8160/8060 and 1500) are locked to vatType 0
   - Posting on 1500 MUST include customer: {"id": CUSTOMER_ID}
   - POST /ledger/voucher with date = payment date

Example: Invoice 10000 EUR at 11.50 NOK/EUR. Paid at 11.20 NOK/EUR.
Disagio = (11.50 - 11.20) × 10000 = 3000 NOK loss
POST /ledger/voucher {"date":"2026-03-22", "description":"Disagio", "postings":[
  {"row":1, "account":{"id":ACCT_8160_ID}, "amountGross":3000, "amountGrossCurrency":3000, "vatType":{"id":0}},
  {"row":2, "account":{"id":ACCT_1500_ID}, "amountGross":-3000, "amountGrossCurrency":-3000, "vatType":{"id":0}, "customer":{"id":CUST_ID}}
]}
""",

    "credit_note": """\
## Credit Note (kreditnota / Gutschrift / nota de crédito)
Full flow (sandbox is EMPTY — must create everything from scratch):

1. POST /customer — include ALL details from prompt:
   {{"name":"Company AS", "isCustomer":true, "email":"x@y.no", "organizationNumber":"123456789", "phoneNumber":"12345678", "postalAddress":{{"addressLine1":"Street 1","postalCode":"1234","city":"Oslo"}}}}
   CRITICAL: Include organizationNumber, phoneNumber, email, postalAddress if mentioned! Missing fields = lost points!

2. POST /order {{"customer":{{"id":CUST_ID}}, "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD", "orderLines":[{{"description":"Product/Service Name", "count":1, "unitPriceExcludingVatCurrency":AMOUNT}}]}}
   - Use the EXACT product/service description from the prompt
   - Amount should be the price EXCLUDING VAT (ekskl. mva)

3. PUT /order/ORDER_ID/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD&sendToCustomer=false
   - invoiceDueDate = invoiceDate + 14 days if not specified

4. PUT /invoice/INV_ID/:createCreditNote?date=YYYY-MM-DD&comment=REASON
   - date is REQUIRED as query param (use today's date if not specified)
   - comment: use the reason from the prompt (e.g. "Kreditering", "Reklamasjon", "Gutschrift")
   - Returns the credit note invoice object

IMPORTANT: The sandbox starts EMPTY. There is NO pre-existing invoice. You MUST create customer → order → invoice → credit note.
SCORING: The scorer checks EVERY field — customer name, org number, phone, email, address, product description, amount, credit note date, comment. Missing ANY = lost points.
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
## Salary (lønn) — "Kjør lønn" / "Run payroll"
The scoring checks: employee exists, salary transaction exists, correct amounts are booked.

COMPLETE STEPS (all required, in order):

1. GET /department?fields=id,name&count=1 — need department ID for employee

2. POST /employee {{"firstName":"X", "lastName":"Y", "email":"x@y.no", "dateOfBirth":"1990-01-01", "userType":"STANDARD", "department":{{"id":DEPT_ID}}}}
   - dateOfBirth is REQUIRED — salary module will fail without it
   - department is REQUIRED — employee must belong to a department

3. PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=EMP_ID&template=ALL_PRIVILEGES
   - Grants the employee access to salary module features
   - Must be done BEFORE creating employment

4. POST /employee/employment {{"employee":{{"id":EMP_ID}}, "startDate":"YYYY-MM-DD"}}
   - ONLY employee.id and startDate — NO other fields
   - startDate should be first day of the month or employment start date from task

5. POST /employee/employment/details {{"employment":{{"id":EMPL_ID}}, "date":"YYYY-MM-DD", "annualSalary": MONTHLY_SALARY * 12, "percentageOfFullTimeEquivalent": 100.0}}
   - annualSalary = the monthly base salary × 12 (e.g. 40850 × 12 = 490200)
   - annualSalary goes HERE on employment/details, NOT on the employee object
   - date = same as the employment startDate
   - percentageOfFullTimeEquivalent = 100.0 (NOT 1.0 — 100.0 means full time)

6. POST /salary/transaction {{"year":YYYY, "month":MM}}
   - year/month = the payroll period from the prompt (e.g. "denne måneden" = current month)
   - NOTE: employee is NOT a field on salary/transaction! Employee is linked via payslips automatically.
   - Employment + employment details MUST exist BEFORE this step
   - FORBIDDEN fields: salaryLines, salaryTransaction, line, amount, baseSalary, payslips, employee

7. GET /salary/payslip?employeeId=EMP_ID&yearFrom=YYYY&monthFrom=MM&fields=id,employee,year,month
   - Retrieves the payslip created by the salary transaction
   - Need the PAYSLIP_ID for bonus, calculate, and close steps

8. For BONUS or one-time additions via payslip specification:
   a. GET /salary/type?fields=id,number,name — find the correct salary type for bonus
      (look for types like "Bonus", "Engangsutbetaling", or similar)
   b. POST /salary/payslip/PAYSLIP_ID/specification {{"rate":BONUS_AMOUNT, "count":1, "salaryType":{{"id":SALARY_TYPE_ID}}}}
      - rate = the bonus amount
      - count = 1
      - salaryType.id = the ID from GET /salary/type

9. PUT /salary/payslip/PAYSLIP_ID/:calculate
   - Calculates the payslip (tax, deductions, net pay)
   - Must be done BEFORE closing

10. PUT /salary/payslip/PAYSLIP_ID/:close
    - Finalizes the payslip — books the salary to the ledger
    - This is what creates the actual accounting entries

CRITICAL: dateOfBirth is REQUIRED on employee for salary module to work
CRITICAL: Employee MUST have a department
CRITICAL: Employment + employment details MUST exist BEFORE creating salary transaction
CRITICAL: annualSalary goes on employment/details, NOT on the employee object
CRITICAL: percentageOfFullTimeEquivalent = 100.0 (NOT 1.0)
CRITICAL: Steps 9 (calculate) and 10 (close) are needed to finalize payroll

FORBIDDEN endpoints: /salary/transaction/line (405), /salary/payslip/ID/line (404), PUT /salary/payslip (405)

FALLBACK for bonus if payslip specification fails — book a voucher directly:
   GET /ledger/account?number=5000&fields=id (salary expense)
   GET /ledger/account?number=2910&fields=id (salary payable/påløpt lønn)
   POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Engangsbonus for Employee Name",
     "postings":[
       {{"row":1, "account":{{"id":5000_ID}}, "amountGross":BONUS_AMOUNT, "amountGrossCurrency":BONUS_AMOUNT, "vatType":{{"id":0}}, "employee":{{"id":EMP_ID}}}},
       {{"row":2, "account":{{"id":2910_ID}}, "amountGross":-BONUS_AMOUNT, "amountGrossCurrency":-BONUS_AMOUNT, "vatType":{{"id":0}}, "employee":{{"id":EMP_ID}}}}
     ]}}
   Account 5000 and 2910 postings MUST include employee: {{"id": EMP_ID}}
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
CRITICAL: You MUST use POST /supplierInvoice (NOT /ledger/voucher). The scoring checks for a SupplierInvoice entity.

Steps:
1. POST /supplier {{"name":"X", "organizationNumber":"X"}} — create supplier
2. GET /ledger/account?number=EXPENSE_ACCT&fields=id (expense account, e.g. 6500, 6700, 7100)
3. GET /ledger/account?number=2400&fields=id (accounts payable)
4. POST /supplierInvoice with this EXACT body structure:
{{
  "invoiceNumber": "INV-XXX",
  "invoiceDate": "YYYY-MM-DD",
  "invoiceDueDate": "YYYY-MM-DD",
  "amountCurrency": GROSS_AMOUNT,
  "supplier": {{"id": SUPPLIER_ID}},
  "voucher": {{
    "date": "YYYY-MM-DD",
    "description": "Leverandørfaktura INV-XXX fra SupplierName",
    "postings": [
      {{"row": 1, "account": {{"id": EXPENSE_ACCT_ID}}, "amountGross": AMOUNT, "amountGrossCurrency": AMOUNT, "vatType": {{"id": 1}}, "supplier": {{"id": SUPPLIER_ID}}}},
      {{"row": 2, "account": {{"id": AP_2400_ID}}, "amountGross": -AMOUNT, "amountGrossCurrency": -AMOUNT, "vatType": {{"id": 0}}, "supplier": {{"id": SUPPLIER_ID}}}}
    ]
  }}
}}

CRITICAL RULES:
- invoiceDueDate is REQUIRED (use invoiceDate + 30 days if not specified)
- amountCurrency = the GROSS amount as a POSITIVE number (including VAT)
- Each posting MUST include supplier: {{"id": SUPPLIER_ID}}
- FORBIDDEN fields on /supplierInvoice: orderDate, deliveryDate, dueDate, amount, orderLines (cause 500!)
- amountGross = the FULL amount (including VAT). Tripletex calculates VAT split automatically.
- Postings MUST sum to zero (expense positive, AP negative)
- vatType mapping: expense 4xxx/6xxx → {{"id":1}} (25%), AP 2400 → {{"id":0}}
- 7xxx accounts → vatType {{"id":0}} (not 1!)
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
IMPORTANT: This creates a VOUCHER (POST /ledger/voucher), NOT a supplier invoice! Do NOT use /supplierInvoice!

The task gives you a receipt (text in prompt, or image/PDF attachment). You must:
1. Read the receipt text from the prompt — extract: item name, amount (inkl/eksl MVA), date, vendor
2. Determine the correct expense account based on what was purchased:
   - Computer peripherals/mus/tastatur/skjerm/IT → 6540 (Inventar og utstyr)
   - Hotel/overnatting → 7140 (Reise og diett)
   - Restaurant/mat/food → 6340 (Serveringskostnader)
   - Office supplies/kontor/papir → 6540 (Inventar og utstyr)
   - Phone/telefon → 6900 (Telefon)
   - Transport/taxi/bus → 7140 (Reise og diett)
   - Parking → 7130 (Parkering)
   - Flight/fly → 7140 (Reise og diett)
   - Software/lisens/subscription → 6540 (Inventar og utstyr)
   - Cleaning/renhold → 6300 (Renhold)
3. Determine VAT type ID:
   - 1 = incoming 25% (standard goods/services — electronics, office supplies, most purchases)
   - 11 = incoming 15% (food/groceries)
   - 13 = incoming 12% (transport/hotel)
   - 0 = no VAT (exempt)
4. If department is specified in the task: GET /department?name=X or POST /department if not found
5. POST /ledger/voucher with postings

EXACT STEPS:
1. GET /ledger/account?number=EXPENSE_ACCT&fields=id (e.g. 6540 for equipment)
2. GET /ledger/account?number=1920&fields=id (bank account — credit side)
3. If department specified: GET /department?name=X&fields=id,name
   - If department NOT found: POST /department {{"name":"X", "departmentNumber":N}}
4. POST /ledger/voucher {{
     "date":"YYYY-MM-DD",
     "description":"Kvittering: ITEM_NAME",
     "postings":[
       {{"row":1, "account":{{"id":EXPENSE_ACCT_ID}}, "amountGross":GROSS_AMOUNT, "amountGrossCurrency":GROSS_AMOUNT, "vatType":{{"id":VAT_TYPE_ID}}, "department":{{"id":DEPT_ID}}}},
       {{"row":2, "account":{{"id":BANK_ACCT_ID}}, "amountGross":-GROSS_AMOUNT, "amountGrossCurrency":-GROSS_AMOUNT, "vatType":{{"id":0}}}}
     ]}}

CRITICAL:
- amountGross = the FULL amount INCLUDING VAT. Tripletex splits VAT automatically based on vatType.
- Account 1920 (bank) is ALWAYS vatType 0.
- The expense account posting gets the incoming VAT type (1, 11, or 13).
- If department is mentioned, include department:{{"id":DEPT_ID}} on the EXPENSE posting (row 1), NOT on the bank posting.
- Do NOT use /supplierInvoice! This is a simple voucher booking.
- STOP after creating the voucher. Do not create anything else.
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
5. POST /employee/employment/details {{"employment":{{"id":EMPLOYMENT_ID}}, "date":"START_DATE",
     "annualSalary":MONTHLY*12, "percentageOfFullTimeEquivalent":100.0}}
   - annualSalary = monthly salary × 12
   - percentageOfFullTimeEquivalent = 100.0 for full-time (NOT 1.0!)
6. If role/entitlement mentioned: PUT /employee/entitlement/:grantEntitlementsByTemplate?employeeId=EMP_ID&template=ALL_PRIVILEGES

CRITICAL: nationalIdentityNumber is a valid field on Employee. Include it if found in PDF.
CRITICAL: Do NOT include employmentType or percentageOfFullTimeEquivalent on /employee/employment — only employee.id and startDate. These go on /employment/details!
CRITICAL: Phone field is phoneNumberMobile on Employee (NOT phoneNumber).
CRITICAL: bankAccountNumber — include if found in PDF (employee bank account for salary).
""",

    "bank_reconciliation_csv": """\
## Bank Reconciliation from CSV (bankavstemminger)
SANDBOX STARTS EMPTY! There are NO pre-existing invoices, customers, or suppliers!

### Step 0: Parse the CSV
The task includes a CSV bank statement (may be base64-encoded attachment).
Decode it, then parse rows. Typical columns: date, description/text, amount, reference/KID.
- POSITIVE amounts = incoming payments (customer paid us)
- NEGATIVE amounts = outgoing payments (we paid a supplier/expense)

### Step 1: Look up accounts and payment types ONCE (cache these!)
- GET /invoice/paymentType?fields=id,description → cache the first paymentTypeId
- GET /ledger/account?number=1920&fields=id → bank account ID
- GET /ledger/account?number=2400&fields=id → accounts payable (leverandørgjeld) ID
- GET /ledger/account?number=1500&fields=id → accounts receivable (kundefordringer) ID

### Step 2: For each INCOMING payment (positive amount) — Customer invoice flow
1. POST /customer {{"name":"CustomerName", "isCustomer":true}}
   - Use the description/text field to derive customer name
2. POST /order {{"customer":{{"id":CUST_ID}}, "orderDate":"DATE", "deliveryDate":"DATE",
   "orderLines":[{{"description":"Payment", "count":1, "unitPriceExcludingVatCurrency":AMOUNT_EX_VAT, "vatType":{{"id":3}}}}]}}
   - deliveryDate is REQUIRED — set same as orderDate
   - unitPriceExcludingVatCurrency = amount / 1.25 (if VAT applies)
   - vatType 3 = outgoing 25% MVA (standard for sales)
3. PUT /order/ORDER_ID/:invoice?invoiceDate=DATE&invoiceDueDate=DATE&sendToCustomer=false
   - invoiceDueDate is REQUIRED — use same date as invoiceDate (already paid)
4. PUT /invoice/INV_ID/:payment?paymentDate=DATE&paymentTypeId=X&paidAmount=AMOUNT&paidAmountCurrency=AMOUNT
   - paidAmount = FULL amount INCLUDING VAT (the CSV amount)
   - paidAmountCurrency = same as paidAmount (both are REQUIRED)
   - All params are QUERY params, NOT body

### Step 3: For each OUTGOING payment (negative amount) — Supplier/expense voucher
1. POST /ledger/voucher {{"date":"DATE", "description":"DESC from CSV",
   "postings":[
     {{"row":1, "account":{{"id":AP_2400_ID}}, "amountGross":ABS_AMOUNT, "amountGrossCurrency":ABS_AMOUNT, "vatType":{{"id":0}}}},
     {{"row":2, "account":{{"id":BANK_1920_ID}}, "amountGross":-ABS_AMOUNT, "amountGrossCurrency":-ABS_AMOUNT, "vatType":{{"id":0}}}}
   ]}}
   - Debit 2400 (leverandørgjeld), Credit 1920 (bank)
   - ALL balance sheet accounts (1xxx, 2xxx) use vatType 0
   - ABS_AMOUNT = absolute value of the negative CSV amount

### Step 4: Unmatched/other items → General ledger voucher
For rows that don't fit customer payment or supplier payment patterns:
- POST /ledger/voucher with appropriate account postings
- Bank fees → Debit 7770 (bankgebyr), Credit 1920 (bank)
- Interest income → Debit 1920 (bank), Credit 8040 (renteinntekt)

### CRITICAL RULES:
- Parse ALL rows in the CSV — every single row must be processed
- POSITIVE amount = customer payment (create customer→order→invoice→payment)
- NEGATIVE amount = outgoing payment (create voucher: debit 2400, credit 1920)
- Postings on account 1500 MUST include customer: {{"id": CUSTOMER_ID}}
- Postings on account 2400 MUST include supplier: {{"id": SUPPLIER_ID}} if a supplier exists
- GET /invoice REQUIRES both invoiceDateFrom AND invoiceDateTo params
- Row numbers in postings start from 1 (NEVER 0)
- amountGrossCurrency MUST be included on every posting (same value as amountGross)
- Postings MUST sum to zero
""",

    "ledger_analysis": """\
## Ledger Analysis — Find Top 3 Cost Accounts with Largest Increase Jan→Feb

### Step 1: Get ALL January postings
GET /ledger/posting with params: {{"dateFrom":"2026-01-01","dateTo":"2026-01-31","fields":"account(id,number),amount,date","count":"10000"}}
CRITICAL: endpoint is /ledger/posting (NOT /ledger/voucher/posting). dateFrom AND dateTo are REQUIRED.

### Step 2: Get ALL February postings
GET /ledger/posting with params: {{"dateFrom":"2026-02-01","dateTo":"2026-02-28","fields":"account(id,number),amount,date","count":"10000"}}

### Step 3: Analyze the data
- Group ALL postings by account number
- Sum the "amount" field per account per month
- Keep ONLY cost accounts: account numbers 4000-7999 (4xxx, 5xxx, 6xxx, 7xxx)
- Calculate increase = feb_total - jan_total for each cost account
- Find the 3 accounts with the LARGEST POSITIVE increase

### Step 4: Create a corrective voucher for EACH of the top 3 accounts
For each account, GET /ledger/account?number=ACCT_NUM&fields=id to get account ID, then:
POST /ledger/voucher with body:
{{"date":"2026-02-28","description":"Kostnadsanalyse: Konto ACCT_NUM økning fra jan til feb",
  "postings":[
    {{"row":1,"account":{{"id":COST_ACCT_ID}},"amountGross":INCREASE_AMOUNT,"amountGrossCurrency":INCREASE_AMOUNT,"vatType":{{"id":0}}}},
    {{"row":2,"account":{{"id":BALANCE_ACCT_ID}},"amountGross":-INCREASE_AMOUNT,"amountGrossCurrency":-INCREASE_AMOUNT,"vatType":{{"id":0}}}}
  ]}}
Use account 1920 (bank) or another balance account for the contra entry.

CRITICAL: The "amount" field in postings is what you sum — it can be positive or negative.
CRITICAL: You MUST fetch ALL postings (count=10000), not just a few.
CRITICAL: Do NOT query per-account — get ALL postings in two bulk queries (one per month).
""",

    "supplier_invoice_pdf": """\
## Supplier Invoice from PDF
1. Read the PDF — extract: supplier name, org number, invoice number, date, due date, amount, expense account
2. POST /supplier {{"name":"X", "organizationNumber":"Y"}} (create if not exists)
3. GET /ledger/account?number=EXPENSE_ACCT&fields=id (e.g. 6500, 6700, 7100)
4. GET /ledger/account?number=2400&fields=id (leverandørgjeld/accounts payable)
5. POST /supplierInvoice (NOT /ledger/voucher — scoring checks SupplierInvoice entity!):
   {{"invoiceNumber":"INV-XXX", "invoiceDate":"YYYY-MM-DD", "invoiceDueDate":"YYYY-MM-DD",
     "amountCurrency":FULL_AMOUNT, "supplier":{{"id":SUPPLIER_ID}},
     "voucher":{{"date":"YYYY-MM-DD", "description":"Leverandørfaktura INV-XXX fra SupplierName",
       "postings":[
         {{"row":1, "account":{{"id":EXPENSE_ACCT_ID}}, "amountGross":FULL_AMOUNT, "amountGrossCurrency":FULL_AMOUNT, "vatType":{{"id":1}}, "supplier":{{"id":SUPPLIER_ID}}}},
         {{"row":2, "account":{{"id":AP_ACCT_ID}}, "amountGross":-FULL_AMOUNT, "amountGrossCurrency":-FULL_AMOUNT, "vatType":{{"id":0}}, "supplier":{{"id":SUPPLIER_ID}}}}
       ]}}
   }}

CRITICAL: amountCurrency = POSITIVE gross amount. invoiceDueDate is REQUIRED.
FORBIDDEN fields: orderDate, deliveryDate, dueDate, amount, orderLines (cause 500).
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

# First month_end_closing definition removed — second definition below is authoritative

API_GUIDES["currency_exchange"] = """\
## Currency Exchange / Agio (valutadifferanse)
Task: Sent invoice in foreign currency (e.g. EUR) at exchange rate X NOK/EUR.
Customer paid in EUR. Record payment and post exchange rate gain/loss voucher.

CALCULATION:
- Invoice NOK amount = EUR_amount × invoice_rate (e.g. 11671 × 11.22 = 130,948.62)
- Payment NOK amount = EUR_amount × payment_rate (or same rate if not specified differently)
- Agio difference = Invoice NOK amount - Payment NOK amount
- If difference > 0: exchange rate LOSS (debit 8160). If < 0: exchange rate GAIN (credit 8060).
- If prompt says "exchange rate was X NOK/EUR" and doesn't mention a different payment rate,
  the payment was at a DIFFERENT rate. Read carefully — "record the payment and post exchange rate gain/loss"
  means BOTH the invoice AND payment used the SAME rate, and you just need to book the rounding difference.
  OR the prompt gives two different rates.

Steps:
1. POST /customer {{"name":"X", "organizationNumber":"Y", "isCustomer":true}}
2. GET /currency?code=EUR&fields=id,code,factor (NOT name — causes 400!)
3. POST /order {{"customer":{{"id":CUST_ID}}, "currency":{{"id":EUR_CURRENCY_ID}},
     "orderDate":"YYYY-MM-DD", "deliveryDate":"YYYY-MM-DD",
     "orderLines":[{{"description":"X", "count":1, "unitPriceExcludingVatCurrency":EUR_AMOUNT, "vatType":{{"id":3}}}}]}}
   CRITICAL: unitPriceExcludingVatCurrency is in EUR (the foreign currency), NOT in NOK!
4. PUT /order/ORDER_ID/:invoice?invoiceDate=YYYY-MM-DD&invoiceDueDate=YYYY-MM-DD&sendToCustomer=false
5. GET /invoice?orderIds=ORDER_ID&invoiceDateFrom=YYYY-MM-DD&invoiceDateTo=YYYY-MM-DD&fields=id,invoiceNumber,amount
   (to get the invoice ID)
6. GET /invoice/paymentType?fields=id,description
7. PUT /invoice/INV_ID/:payment?paymentDate=YYYY-MM-DD&paymentTypeId=X&paidAmount=NOK_AMOUNT&paidAmountCurrency=EUR_AMOUNT
   - paidAmount = EUR_amount × payment_rate (NOK amount)
   - paidAmountCurrency = EUR_amount (in foreign currency)
8. Book agio voucher:
   GET /ledger/account?number=8060&fields=id (finansinntekt — for gain)
   GET /ledger/account?number=8160&fields=id (finanskostnad — for loss)
   GET /ledger/account?number=1500&fields=id (kundefordringer)

   AGIO_AMOUNT = abs(invoice_nok - payment_nok)

   If LOSS (invoice_nok > payment_nok — customer paid less in NOK than expected):
   POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Agiotap valutadifferanse",
     "postings":[
       {{"row":1, "account":{{"id":8160_ID}}, "amountGross":AGIO_AMOUNT, "amountGrossCurrency":AGIO_AMOUNT, "vatType":{{"id":0}}}},
       {{"row":2, "account":{{"id":1500_ID}}, "amountGross":-AGIO_AMOUNT, "amountGrossCurrency":-AGIO_AMOUNT, "vatType":{{"id":0}}, "customer":{{"id":CUST_ID}}}}
     ]}}

   If GAIN (payment_nok > invoice_nok):
   POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Agiogevinst valutadifferanse",
     "postings":[
       {{"row":1, "account":{{"id":1500_ID}}, "amountGross":AGIO_AMOUNT, "amountGrossCurrency":AGIO_AMOUNT, "vatType":{{"id":0}}, "customer":{{"id":CUST_ID}}}},
       {{"row":2, "account":{{"id":8060_ID}}, "amountGross":-AGIO_AMOUNT, "amountGrossCurrency":-AGIO_AMOUNT, "vatType":{{"id":0}}}}
     ]}}

CRITICAL: Account 1500 postings MUST include customer: {{"id": CUST_ID}}
CRITICAL: vatType {{"id":0}} for ALL accounts (1500, 8060, 8160 are locked to vatType 0)
"""
API_GUIDES["agio"] = API_GUIDES["currency_exchange"]
API_GUIDES["disagio"] = API_GUIDES["currency_exchange"]
API_GUIDES["valutakurs"] = API_GUIDES["currency_exchange"]
API_GUIDES["exchange_rate"] = API_GUIDES["currency_exchange"]
API_GUIDES["currency"] = API_GUIDES["currency_exchange"]
API_GUIDES["EUR"] = API_GUIDES["currency_exchange"]
API_GUIDES["foreign_currency"] = API_GUIDES["currency_exchange"]
API_GUIDES["valuta"] = API_GUIDES["currency_exchange"]

API_GUIDES["overdue_invoice_reminder"] = """\
## Overdue Invoice + Reminder Fee (purregebyr / forfalt faktura)
SANDBOX IS EMPTY! You must create EVERYTHING from scratch. There are NO existing customers, invoices, or payments.

The task typically says: "Customer X has an overdue invoice for Product Y (amount Z). Book a reminder fee of W kr."
You must: create the customer, create the overdue invoice, optionally register partial payment, then book the reminder fee as a voucher.

COMPLETE FLOW (follow EXACTLY — do NOT create extra orders or invoices):

Step 1: POST /customer — include ALL details from prompt:
  {{"name":"Company AS", "isCustomer":true, "organizationNumber":"123456789", "phoneNumber":"12345678", "email":"x@y.no", "postalAddress":{{"addressLine1":"Street 1","postalCode":"1234","city":"Oslo"}}}}

Step 2: POST /order — use PAST dates so invoice appears overdue:
  {{"customer":{{"id":CUST_ID}}, "orderDate":"2026-01-15", "deliveryDate":"2026-01-15",
   "orderLines":[{{"description":"Product from prompt", "count":1, "unitPriceExcludingVatCurrency":AMOUNT_FROM_PROMPT}}]}}

Step 3: PUT /order/ORDER_ID/:invoice?invoiceDate=2026-01-15&invoiceDueDate=2026-02-15&sendToCustomer=false
  - Use dates 1-2 months in the PAST so the invoice is overdue

Step 4 (only if partial payment mentioned):
  GET /invoice/paymentType?fields=id,description
  PUT /invoice/INV_ID/:payment?paymentDate=DATE&paymentTypeId=X&paidAmount=PARTIAL_AMOUNT

Step 5: Book the reminder fee voucher:
  GET /ledger/account?number=1500&fields=id (Kundefordringer — debit side)
  GET /ledger/account?number=3400&fields=id (Offentlige avgifter/gebyrer — credit side, revenue)
  POST /ledger/voucher {{"date":"2026-03-22", "description":"Purregebyr", "postings":[
    {{"row":1, "account":{{"id":ACCT_1500_ID}}, "amountGross":FEE_AMOUNT, "amountGrossCurrency":FEE_AMOUNT, "vatType":{{"id":0}}, "customer":{{"id":CUST_ID}}}},
    {{"row":2, "account":{{"id":ACCT_3400_ID}}, "amountGross":-FEE_AMOUNT, "amountGrossCurrency":-FEE_AMOUNT, "vatType":{{"id":0}}}}
  ]}}

CRITICAL RULES:
- Account 1500 posting MUST include customer: {{"id": CUST_ID}}
- Account 3400 is locked to vatType 0 — ALWAYS use vatType:{{"id":0}}
- Create ONLY ONE order and ONE invoice. Do NOT create a second order/invoice for the reminder fee.
- The reminder fee is booked as a VOUCHER (POST /ledger/voucher), NOT as a second invoice.
- If the prompt specifies account numbers (e.g. "debet konto 1500, kredit konto 3400"), use those exact accounts.
- STOP after posting the voucher. Do not create anything else.
"""
API_GUIDES["overdue"] = API_GUIDES["overdue_invoice_reminder"]
API_GUIDES["forfalt"] = API_GUIDES["overdue_invoice_reminder"]
API_GUIDES["impaye"] = API_GUIDES["overdue_invoice_reminder"]

API_GUIDES["month_end_closing"] = """\
## Month-End Closing (månedsslutt)
Post these vouchers for the specified month:

### 1. Accrual reversal (periodisering)
Move prepaid amount from balance sheet to expense:
- Debit: expense account (e.g. 6xxx-7xxx) with monthly_amount
- Credit: prepaid account (e.g. 1720) with -monthly_amount
POST /ledger/voucher {"date":"YYYY-MM-DD", "description":"Periodisering", "postings":[
  {"row":1, "account":{"id":EXPENSE_ID}, "amountGross":MONTHLY_AMT, "amountGrossCurrency":MONTHLY_AMT, "vatType":{"id":0}},
  {"row":2, "account":{"id":1720_ID}, "amountGross":-MONTHLY_AMT, "amountGrossCurrency":-MONTHLY_AMT, "vatType":{"id":0}}
]}

### 2. Depreciation (avskrivning)
For each asset: annual_depreciation = asset_value / useful_life_years
Monthly: annual / 12
- Debit: depreciation expense (6010-6020) with monthly_depreciation
- Credit: accumulated depreciation (1209) with -monthly_depreciation
POST /ledger/voucher {"date":"YYYY-MM-DD", "description":"Avskrivning", "postings":[
  {"row":1, "account":{"id":6010_ID}, "amountGross":MONTHLY_DEPR, "amountGrossCurrency":MONTHLY_DEPR, "vatType":{"id":0}},
  {"row":2, "account":{"id":1209_ID}, "amountGross":-MONTHLY_DEPR, "amountGrossCurrency":-MONTHLY_DEPR, "vatType":{"id":0}}
]}

### 3. Salary accrual (lønn)
If salary mentioned: accrue monthly salary
- Debit: salary expense (5000) with salary_amount
- Credit: salary payable (2910) with -salary_amount
- CRITICAL: Account 5000 posting MUST include employee ref! Create employee first if none exists.
POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Lønnsavsetning", "postings":[
  {{"row":1, "account":{{"id":5000_ID}}, "amountGross":SALARY, "amountGrossCurrency":SALARY, "vatType":{{"id":0}}, "employee":{{"id":EMP_ID}}}},
  {{"row":2, "account":{{"id":2910_ID}}, "amountGross":-SALARY, "amountGrossCurrency":-SALARY, "vatType":{{"id":0}}}}
]}}

CRITICAL:
- ALL accounts (1209, 1720, 5000, 2910, 6010 etc.) are locked to vatType 0
- Date should be last day of the month (e.g. 2026-03-31)
- GET /ledger/account?number=XXXX&fields=id for each account first
- Create SEPARATE vouchers for each type (accrual, depreciation, salary)
- Account 5000 postings MUST include employee: {{"id": X}}
- Account 1500 postings MUST include customer: {{"id": X}}
- Account 2400 postings MUST include supplier: {{"id": X}}
"""
API_GUIDES["månedsavslutning"] = API_GUIDES["month_end_closing"]
API_GUIDES["closing"] = API_GUIDES["month_end_closing"]
API_GUIDES["accrual"] = API_GUIDES["month_end_closing"]
API_GUIDES["monatsabschluss"] = API_GUIDES["month_end_closing"]
API_GUIDES["månedsslutt"] = API_GUIDES["month_end_closing"]
API_GUIDES["month_end"] = API_GUIDES["month_end_closing"]
API_GUIDES["month-end"] = API_GUIDES["month_end_closing"]
API_GUIDES["periodisering"] = API_GUIDES["month_end_closing"]
API_GUIDES["clôture mensuelle"] = API_GUIDES["month_end_closing"]
API_GUIDES["cierre mensual"] = API_GUIDES["month_end_closing"]

API_GUIDES["ledger_correction"] = """\
## Ledger Correction / Voucher Correction (feilretting i regnskap)

### Step 1: Find ALL vouchers WITH postings in ONE query
GET /ledger/voucher?dateFrom=2026-01-01&dateTo=2026-02-28&fields=id,date,description,postings(id,account(id,number),amountGross,vatType(id),row)&count=1000
- This returns ALL vouchers with their postings in a single call!
- MUST include dateFrom AND dateTo (both required!)
- Do NOT query vouchers individually — use this bulk query

### Step 2: Identify the 4 types of errors
The prompt tells you exactly what errors to find. Common patterns:
a) **Wrong account number** — posting on account X should be on account Y
b) **Duplicate voucher** — same voucher posted twice
c) **Missing VAT** — expense posted without VAT that should have it
d) **Wrong amount** — posted amount differs from correct amount

### Step 4: Correct each error
For WRONG ACCOUNT: POST a correcting voucher with 2 entries:
  - Reverse the wrong posting (opposite sign on wrong account)
  - Post correct amount on right account
  POST /ledger/voucher {{"date":"YYYY-MM-DD", "description":"Korreksjon - feil konto", "postings":[
    {{"row":1, "account":{{"id":WRONG_ACCT_ID}}, "amountGross":-ORIGINAL_AMT, "amountGrossCurrency":-ORIGINAL_AMT, "vatType":{{"id":0}}}},
    {{"row":2, "account":{{"id":CORRECT_ACCT_ID}}, "amountGross":ORIGINAL_AMT, "amountGrossCurrency":ORIGINAL_AMT, "vatType":{{"id":0}}}}
  ]}}

For DUPLICATE: PUT /ledger/voucher/ID/:reverse?date=YYYY-MM-DD

For MISSING VAT: POST correcting voucher adding the VAT posting

For WRONG AMOUNT: POST correcting voucher for the difference only

CRITICAL: GET /ledger/account?number=XXXX&fields=id for each account first.
CRITICAL: Postings MUST sum to zero. vatType 0 for balance sheet accounts.
CRITICAL: amountGrossCurrency MUST equal amountGross (same value).
"""
API_GUIDES["feilretting"] = API_GUIDES["ledger_correction"]
API_GUIDES["voucher_correction"] = API_GUIDES["ledger_correction"]
API_GUIDES["correction"] = API_GUIDES["ledger_correction"]
API_GUIDES["korreksjon"] = API_GUIDES["ledger_correction"]

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
    # Amount fields needed for ledger analysis, bank reconciliation, etc.
    "amount", "amountGross", "amountGrossCurrency", "amountCurrency",
    "amountOutstanding", "date", "invoiceDate", "invoiceDueDate",
    "invoiceNumber", "customer", "supplier", "account", "vatType",
    "row", "postings", "rate", "code", "factor", "displayName",
})


# ── Agent execution ──────────────────────────────────────────────────

async def tool_agent_solve(
    prompt: str,
    files: list[dict] | None,
    client: TripletexClient,
    deadline: float,
) -> bool:
    """Run the tool-use agent. Returns True if task completed without errors."""

    # Use Pro for complex tasks (better reasoning), Flash for simple ones (faster)
    prompt_lower = prompt.lower()
    is_complex = any(kw in prompt_lower for kw in (
        "reconcil", "avstem", "rapproch", "year-end", "årsoppgjør", "årsoppgjer", "arsoppgjor", "jahresabschluss",
        "month-end", "monatsabschluss", "månedsslutt", "månadsslutt", "cierre", "encerramento", "clôture",
        "feil i hovedbok", "feil i hovudbok", "error in ledger", "fehler", "errores", "erros", "korriger",
        "analyser", "analyse", "analyze", "analise", "kostnadskonto",
        "lifecycle", "livssyklus", "prosjektsyklusen", "ciclo de vida",
        "exchange rate", "valutakurs", "agio", "disagio", "wechselkurs", "taux de change",
        "eur ", "usd ", "gbp ",
        "lønn", "salary", "gehalt", "salaire", "salário", "payroll",
    ))
    model_name = "gemini-2.5-pro" if is_complex else "gemini-2.5-flash"
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
    # Auto-inject relevant guides based on keywords in prompt
    prompt_lower = prompt.lower()
    injected_guides = []
    guide_triggers = [
        (("eur ", "usd ", "gbp ", "exchange rate", "valutakurs", "agio", "disagio", "tipo de cambio", "wechselkurs", "taux de change", "taxa de câmbio"), "currency_exchange"),
        (("month-end", "månedsavslutning", "accrual", "periodisering", "lønnsavsetning", "salary accrual", "encerramento mensal", "monatsabschluss", "cierre mensual"), "month_end_closing"),
        (("lønn", "salary", "payroll", "salario", "gehalt", "salaire", "grunnlønn", "bonus"), "salary"),
        (("leverandørfaktura", "supplier invoice", "fournisseur", "lieferantenrechnung", "factura del proveedor", "fatura do fornecedor"), "supplier_invoice"),
        (("reconcil", "avstemming", "bankutskrift", "bank statement", "csv"), "bank_reconciliation_csv"),
        (("lifecycle", "livssyklus", "ciclo de vida", "prosjektsyklusen"), "project_lifecycle"),
        (("feil i hovedbok", "error in ledger", "errors in the general", "discovered errors", "korriger", "correct the errors", "rette opp", "feil i bilag", "fehler im hauptbuch", "errores en el libro", "erros no livro", "erreurs dans le grand"), "ledger_correction"),
        (("årsoppgjør", "årsoppgjer", "arsoppgjor", "year-end", "jahresabschluss", "cierre anual", "encerramento anual", "clôture annuelle"), "year_end_closing"),
        (("analyser", "analyse", "analyze", "analise", "analysez", "kostnadskonto", "kostnadsauke", "cost account"), "ledger_analysis"),
        (("arbeidskontrakt", "arbeitsvertrag", "angebotsschreiben", "employment contract", "offer letter", "contrato de trabajo", "contrat de travail", "tilbudsbrev"), "employment_contract_pdf"),
        (("overdue", "forfalt", "purregebyr", "reminder fee", "impayé", "vencida", "überfällig", "en mora", "Mahngebühr"), "overdue_invoice_reminder"),
        (("kvittering", "receipt", "recibo", "quittung", "reçu", "bokført på avdeling", "bokfort pa avdeling", "utgiftskonto"), "receipt_voucher"),
        (("kreditnota", "credit note", "gutschrift", "nota de crédito", "nota de credito", "note de crédit", "reklamert", "reklamiert", "reklamasjon"), "credit_note"),
    ]
    for keywords, guide_name in guide_triggers:
        if any(kw in prompt_lower for kw in keywords):
            guide = API_GUIDES.get(guide_name, "")
            if guide:
                injected_guides.append(guide)

    guide_text = ""
    if injected_guides:
        guide_text = "\n\nRELEVANT API GUIDES (pre-loaded for this task):\n" + "\n---\n".join(injected_guides) + "\n\n"

    parts.append(Part.from_text(f"Execute this accounting task:\n\n{prompt}{guide_text}"))

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
                    # FATAL: 403 = expired token, abort immediately
                    if status == 403:
                        logger.error("403 Forbidden — token expired, aborting tool agent")
                        return had_errors
                summary = _compact_response(data, ok, path=path)
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


def _compact_response(data: dict, ok: bool, max_len: int = 1500, path: str = "") -> dict:
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
                # For ledger posting queries, return ALL results (needed for summing)
                needs_all = "/ledger/posting" in path
                limit = len(vals) if needs_all else 10
                compact_list = []
                for item in vals[:limit]:
                    if isinstance(item, dict):
                        compact_list.append({k: v for k, v in item.items() if k in _LIST_ESSENTIAL_FIELDS})
                    else:
                        compact_list.append(item)
                truncation_note = f"(showing {min(limit, len(vals))}/{len(vals)} results)" if len(vals) > limit else ""
                result = {"ok": True, "count": len(vals), "values": compact_list}
                if truncation_note:
                    result["note"] = truncation_note
                return result

    return {"ok": True, "data": json.dumps(data, ensure_ascii=False, default=str)[:max_len]}

"""Pre-built plan templates for Tripletex task types.

Each template defines:
- description: What this task type does
- relevant_schemas: Which entity schemas to include in Stage 2 context
- steps: Ordered API calls with {{placeholder}} values for LLM to fill
- extract_fields: Fields the LLM must extract from the prompt
- optimal_calls: Minimum API calls for perfect execution (for efficiency scoring)
"""

TEMPLATES: dict[str, dict] = {

    # ===== EMPLOYEES =====

    "create_employee": {
        "description": "Create an employee, optionally assign a role/entitlement. If departments exist, include department in body.",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["firstName", "lastName", "email", "dateOfBirth", "phoneNumberMobile", "role"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/department",
                "params": {"fields": "id,name", "count": 1},
            },
            {
                "method": "POST",
                "path": "/employee",
                "body": {
                    "firstName": "{{firstName}}",
                    "lastName": "{{lastName}}",
                    "email": "{{email}}",
                    "dateOfBirth": "{{dateOfBirth}}",
                    "phoneNumberMobile": "{{phoneNumberMobile}}",
                    "userType": "STANDARD",
                    "department": {"id": "$step_0.values[0].id"},
                },
            },
        ],
        "conditional_steps": {
            "if_role": {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {"employeeId": "$step_1.id", "template": "{{role}}"},
            },
        },
    },

    "update_employee": {
        "description": "Update an existing employee's details (phone, email, address, etc.)",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["search_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"firstName": "{{search_firstName}}", "lastName": "{{search_lastName}}", "fields": "id,firstName,lastName,email,phoneNumberMobile,version"},
            },
            {
                "method": "PUT",
                "path": "/employee/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== CUSTOMERS =====

    "create_customer": {
        "description": "Create a customer with contact details",
        "relevant_schemas": ["Customer"],
        "extract_fields": ["name", "email", "organizationNumber", "phoneNumber", "isSupplier"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{name}}",
                    "isCustomer": True,
                    "email": "{{email}}",
                    "phoneNumber": "{{phoneNumber}}",
                    "organizationNumber": "{{organizationNumber}}",
                },
            },
        ],
    },

    # ===== PRODUCTS =====

    "create_product": {
        "description": "Create a product with price and VAT settings",
        "relevant_schemas": ["Product"],
        "extract_fields": ["name", "number", "priceExcludingVatCurrency", "priceIncludingVatCurrency", "description"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/product",
                "body": {
                    "name": "{{name}}",
                    "number": "{{number}}",
                    "priceExcludingVatCurrency": "{{price}}",
                    "description": "{{description}}",
                },
            },
        ],
    },

    # ===== INVOICING =====

    "create_invoice": {
        "description": "Create an invoice: customer -> order with orderLines -> invoice. NOTE: Company must have bankAccountNumber registered. If 422 about 'bankkontonummer', the sandbox is not properly set up.",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate", "customer_email", "orderDate", "deliveryDate"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                },
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_0.id"},
                    "orderDate": "{{orderDate}}",
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_1.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "invoiceDueDate": "{{invoiceDueDate}}",
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "create_invoice_existing_customer": {
        "description": "Create invoice for an existing customer (search by name first)",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_0.values[0].id"},
                    "orderDate": "{{orderDate}}",
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_1.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "invoiceDueDate": "{{invoiceDueDate}}",
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "register_payment": {
        "description": "Register a payment on an existing invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "amount", "paymentDate"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
            },
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_0.values[0].id",
                    "paidAmount": "{{amount}}",
                },
            },
        ],
    },

    "register_payment_by_search": {
        "description": "Register a payment on an invoice found by searching (by invoice number or customer)",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoiceNumber", "customer_name", "amount", "paymentDate"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/invoice",
                "params": {"invoiceNumber": "{{invoiceNumber}}", "fields": "id,invoiceNumber,amount"},
            },
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
            },
            {
                "method": "PUT",
                "path": "/invoice/$step_0.values[0].id/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_1.values[0].id",
                    "paidAmount": "{{amount}}",
                },
            },
        ],
    },

    "create_credit_note": {
        "description": "Create a credit note for an existing invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "date", "comment"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:createCreditNote",
                "params": {
                    "date": "{{date}}",
                    "comment": "{{comment}}",
                },
            },
        ],
    },

    "send_invoice": {
        "description": "Send an invoice to the customer",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "sendType", "email"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:send",
                "params": {
                    "sendType": "{{sendType}}",
                },
            },
        ],
    },

    # ===== TRAVEL EXPENSES =====

    "create_travel_expense": {
        "description": (
            "Register a travel expense report. Steps:\n"
            "1. GET /employee to find employee ID\n"
            "2. POST /travelExpense with travelDetails (dates, destination, purpose)\n"
            "3. If costs are mentioned: GET /travelExpense/costCategory, GET /travelExpense/paymentType, GET /ledger/vatType, GET /currency?code=NOK\n"
            "4. For EACH cost: POST /travelExpense/cost with the EXACT fields listed below.\n"
            "\n"
            "CRITICAL: POST /travelExpense/cost REQUIRED fields:\n"
            "  - travelExpense: {\"id\": <travel_expense_id>}\n"
            "  - vatType: {\"id\": <vat_type_id>}  (REQUIRED! GET /ledger/vatType first)\n"
            "  - paymentType: {\"id\": <payment_type_id>}\n"
            "  - amountCurrencyIncVat: <number> (the cost amount INCLUDING VAT)\n"
            "  - date: \"YYYY-MM-DD\"\n"
            "OPTIONAL fields:\n"
            "  - currency: {\"id\": <currency_id>}\n"
            "  - costCategory: {\"id\": <cost_category_id>}\n"
            "  - comments: \"string\" (use this for any description/note about the cost)\n"
            "  - rate: <number>\n"
            "  - amountNOKInclVAT: <number>\n"
            "  - isChargeable: boolean\n"
            "  - category: \"string\"\n"
            "\n"
            "FORBIDDEN fields (DO NOT USE — will cause 422 error):\n"
            "  - amount (WRONG — use amountCurrencyIncVat)\n"
            "  - title (WRONG — does not exist)\n"
            "  - description (WRONG — use comments)\n"
            "  - name (WRONG — does not exist)\n"
            "  - rateCurrency (WRONG — not a valid field)\n"
            "  - count (WRONG — not a valid field)"
        ),
        "relevant_schemas": ["TravelExpense", "TravelDetails", "TravelExpenseCost"],
        "extract_fields": ["departureDate", "returnDate", "departureFrom", "destination", "purpose", "costs", "isDayTrip", "isForeignTravel", "title"],
        "optimal_calls": 7,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
            },
            {
                "method": "POST",
                "path": "/travelExpense",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "travelDetails": {
                        "departureDate": "{{departureDate}}",
                        "returnDate": "{{returnDate}}",
                        "departureFrom": "{{departureFrom}}",
                        "destination": "{{destination}}",
                        "purpose": "{{purpose}}",
                        "isDayTrip": "{{isDayTrip}}",
                        "isForeignTravel": "{{isForeignTravel}}",
                    },
                    "title": "{{title}}",
                },
            },
            {
                "method": "GET",
                "path": "/travelExpense/costCategory",
                "params": {"fields": "id,description"},
                "note": "Only include if costs are mentioned. Match category to cost type (e.g. 'Taxi' for taxi).",
            },
            {
                "method": "GET",
                "path": "/travelExpense/paymentType",
                "params": {"fields": "id,description"},
                "note": "Only include if costs are mentioned.",
            },
            {
                "method": "GET",
                "path": "/ledger/vatType",
                "params": {"fields": "id,name,number"},
                "note": "REQUIRED for costs. Get VAT types to find correct vatType ID. Use VAT type 0 (exempt) if unsure.",
            },
            {
                "method": "GET",
                "path": "/currency",
                "params": {"code": "NOK", "fields": "id,code"},
                "note": "Only include if costs are mentioned.",
            },
            {
                "method": "POST",
                "path": "/travelExpense/cost",
                "body": {
                    "travelExpense": {"id": "$step_1.id"},
                    "vatType": {"id": "$step_4.values[0].id"},
                    "paymentType": {"id": "$step_3.values[0].id"},
                    "currency": {"id": "$step_5.values[0].id"},
                    "costCategory": {"id": "{{matched_category_id_from_step_2}}"},
                    "amountCurrencyIncVat": "{{cost_amount}}",
                    "date": "{{departureDate}}",
                    "comments": "{{cost_description_if_any}}",
                },
                "note": "One POST per cost item. REQUIRED: travelExpense, vatType, paymentType, amountCurrencyIncVat, date. FORBIDDEN: amount, title, description, name, rateCurrency, count.",
            },
        ],
    },

    "delete_travel_expense": {
        "description": "Delete a travel expense report",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "DELETE",
                "path": "/travelExpense/{{travel_expense_id}}",
            },
        ],
    },

    "deliver_travel_expense": {
        "description": "Deliver (submit) a travel expense for approval",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/travelExpense/:deliver",
                "params": {"id": "{{travel_expense_id}}"},
            },
        ],
    },

    "approve_travel_expense": {
        "description": "Approve a travel expense",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/travelExpense/:approve",
                "params": {"id": "{{travel_expense_id}}"},
            },
        ],
    },

    # ===== PROJECTS =====

    "create_project": {
        "description": "Create a project linked to a customer. Must set projectManager.",
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["name", "customer_name", "startDate", "endDate", "isInternal", "projectManager", "description"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
            },
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                },
            },
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "description": "{{project_description}}",
                    "customer": {"id": "$step_1.id"},
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "isInternal": False,
                    "projectManager": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    "create_project_existing_customer": {
        "description": "Create a project linked to an existing customer (search by name first). Must set projectManager.",
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["project_name", "customer_name", "startDate", "endDate", "description", "projectManager"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
            },
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "description": "{{project_description}}",
                    "customer": {"id": "$step_1.values[0].id"},
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "isInternal": False,
                    "projectManager": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    "create_internal_project": {
        "description": "Create an internal project (no customer). Must set projectManager.",
        "relevant_schemas": ["Project"],
        "extract_fields": ["name", "startDate", "endDate", "description"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
            },
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "description": "{{project_description}}",
                    "isInternal": True,
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "projectManager": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    # ===== DEPARTMENTS =====

    "create_department": {
        "description": "Create a department",
        "relevant_schemas": ["Department"],
        "extract_fields": ["name", "departmentNumber", "departmentManager"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/department",
                "body": {
                    "name": "{{name}}",
                    "departmentNumber": "{{departmentNumber}}",
                    "departmentManager": {"id": "{{departmentManagerId}}"},
                },
            },
        ],
    },

    # ===== SUPPLIERS =====

    "create_supplier": {
        "description": "Create a supplier",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["name", "organizationNumber", "email", "phoneNumber"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{name}}",
                    "email": "{{email}}",
                    "phoneNumber": "{{phoneNumber}}",
                    "organizationNumber": "{{organizationNumber}}",
                },
            },
        ],
    },

    # ===== UPDATE SUPPLIER =====

    "update_supplier": {
        "description": "Update an existing supplier's details",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["supplier_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/supplier",
                "params": {"name": "{{supplier_name}}", "fields": "id,name,version"},
            },
            {
                "method": "PUT",
                "path": "/supplier/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== UPDATE DEPARTMENT =====

    "update_department": {
        "description": "Update an existing department's details",
        "relevant_schemas": ["Department"],
        "extract_fields": ["department_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/department",
                "params": {"name": "{{department_name}}", "fields": "id,name,departmentNumber,version"},
            },
            {
                "method": "PUT",
                "path": "/department/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== UPDATE PRODUCT =====

    "update_product": {
        "description": "Update an existing product's details",
        "relevant_schemas": ["Product"],
        "extract_fields": ["product_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/product",
                "params": {"name": "{{product_name}}", "fields": "id,name,number,version"},
            },
            {
                "method": "PUT",
                "path": "/product/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== CONTACTS =====

    "create_contact": {
        "description": "Create a contact person for a customer",
        "relevant_schemas": ["Contact", "Customer"],
        "extract_fields": ["firstName", "lastName", "email", "phoneNumber", "customer_name"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/contact",
                "body": {
                    "firstName": "{{firstName}}",
                    "lastName": "{{lastName}}",
                    "email": "{{email}}",
                    "phoneNumber": "{{phoneNumber}}",
                    "customer": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    # ===== LEDGER / VOUCHERS =====

    "create_voucher": {
        "description": (
            "Create a ledger voucher with postings. IMPORTANT: account numbers (e.g. 1920) are NOT IDs — "
            "you must first GET /ledger/account?number=X to find the real account ID.\n"
            "For vouchers with MORE than 2 accounts, add additional GET /ledger/account steps. "
            "Each posting needs the account ID from the GET response. If parsing a file, "
            "each line in the file becomes a posting — parse EVERY line.\n"
            "CRITICAL posting format: each posting is {\"account\": {\"id\": <account_id>}, \"amountGross\": <amount>}. "
            "Positive amountGross = debit, negative = credit. Do NOT include 'row', 'guiRow', or any other fields — "
            "they are system-generated and will cause a 422 error."
        ),
        "relevant_schemas": ["Voucher", "Posting", "Account"],
        "extract_fields": ["date", "description", "postings_with_account_numbers"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{debit_account_number}}", "fields": "id,number,name"},
                "note": "Add one GET step per unique account number. For 3+ accounts, add more GET steps.",
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{credit_account_number}}", "fields": "id,number,name"},
            },
            {
                "method": "POST",
                "path": "/ledger/voucher",
                "body": {
                    "date": "{{date}}",
                    "description": "{{description}}",
                    "postings": [
                        {"account": {"id": "$step_0.values[0].id"}, "amountGross": "{{debit_amount}}"},
                        {"account": {"id": "$step_1.values[0].id"}, "amountGross": "-{{credit_amount}}"},
                    ],
                },
                "note": "Each posting ONLY has 'account.id' and 'amountGross'. No 'row', 'guiRow', or other fields.",
            },
        ],
    },

    "reverse_voucher": {
        "description": "Reverse a voucher",
        "relevant_schemas": ["Voucher"],
        "extract_fields": ["voucher_id", "date"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/ledger/voucher/{{voucher_id}}/:reverse",
                "params": {"date": "{{date}}"},
            },
        ],
    },

    # ===== CORRECTIONS =====

    "delete_entity": {
        "description": "Delete an entity by type and ID",
        "relevant_schemas": [],
        "extract_fields": ["entity_type", "entity_id"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "DELETE",
                "path": "/{{entity_type}}/{{entity_id}}",
            },
        ],
    },

    # ===== SUPPLIER INVOICES =====

    "create_supplier_invoice": {
        "description": "Create a supplier invoice (incoming invoice from a supplier). Requires a supplier, an invoice date, due date, and voucher postings.",
        "relevant_schemas": ["Supplier", "Voucher", "Posting"],
        "extract_fields": ["supplier_name", "invoiceNumber", "invoiceDate", "dueDate", "amount", "account_number", "description"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{supplier_name}}",
                },
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{expense_account_number}}", "fields": "id,number,name"},
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "2400", "fields": "id,number,name"},
                "note": "2400 = leverandørgjeld (accounts payable)",
            },
            {
                "method": "POST",
                "path": "/supplierInvoice",
                "body": {
                    "invoiceNumber": "{{invoiceNumber}}",
                    "invoiceDate": "{{invoiceDate}}",
                    "supplier": {"id": "$step_0.id"},
                    "dueDate": "{{dueDate}}",
                    "voucher": {
                        "date": "{{invoiceDate}}",
                        "description": "{{description}}",
                        "postings": [
                            {"account": {"id": "$step_1.values[0].id"}, "amountGross": "{{amount}}"},
                            {"account": {"id": "$step_2.values[0].id"}, "amountGross": "-{{amount}}"},
                        ],
                    },
                },
            },
        ],
    },

    # ===== PURCHASE ORDERS =====

    "create_purchase_order": {
        "description": "Create a purchase order to a supplier",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["supplier_name", "deliveryDate", "orderLines", "ourContact"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{supplier_name}}",
                },
            },
            {
                "method": "POST",
                "path": "/purchaseOrder",
                "body": {
                    "supplier": {"id": "$step_0.id"},
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
        ],
    },

    # ===== BANK RECONCILIATION =====

    "bank_reconciliation": {
        "description": (
            "Create a bank reconciliation. This is a COMPLEX multi-step process:\n"
            "1. GET /bank to find the bank account ID (look for matching accountNumber)\n"
            "2. POST /bank/reconciliation to create the reconciliation period (MUST include dateFrom, dateTo, type=MANUAL)\n"
            "3. For each transaction: POST /bank/reconciliation/match or create vouchers for unmatched items\n"
            "4. If a bank statement (CSV/file) is attached, use POST /bank/statement/import to import it first\n"
            "5. Unmatched transactions may need manual vouchers via POST /ledger/voucher\n"
            "IMPORTANT: The accounting period must be open for the reconciliation date range. "
            "Use type=MANUAL for manual reconciliation. dateFrom and dateTo must be within the same open period.\n"
            "If the task specifies a closing balance, the sum of matched transactions must equal it."
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date_from", "date_to", "bank_account_number", "transactions", "closing_balance"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/bank",
                "params": {"fields": "id,accountNumber,name"},
                "note": "Find the bank account. Match by accountNumber if specified in the task.",
            },
            {
                "method": "POST",
                "path": "/bank/reconciliation",
                "body": {
                    "account": {"id": "$step_0.values[0].id"},
                    "type": "MANUAL",
                    "dateFrom": "{{date_from}}",
                    "dateTo": "{{date_to}}",
                },
                "note": "dateFrom is required — must be within an open accounting period.",
            },
        ],
    },

    # ===== TIMESHEET =====

    "create_timesheet_entry": {
        "description": "Register hours/timesheet entry for an employee on a project/activity",
        "relevant_schemas": ["Employee", "Project", "Activity"],
        "extract_fields": ["employee_name", "project_name", "activity_name", "date", "hours", "comment"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
            },
            {
                "method": "GET",
                "path": "/project",
                "params": {"name": "{{project_name}}", "fields": "id,name"},
            },
            {
                "method": "GET",
                "path": "/activity",
                "params": {"fields": "id,name"},
            },
            {
                "method": "POST",
                "path": "/timesheet/entry",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "project": {"id": "$step_1.values[0].id"},
                    "activity": {"id": "$step_2.values[0].id"},
                    "date": "{{date}}",
                    "hours": "{{hours}}",
                    "comment": "{{comment}}",
                },
            },
        ],
    },

    # ===== OPENING BALANCE =====

    "create_opening_balance": {
        "description": (
            "Set opening balance entries for the company. Each entry has an account and an amount.\n"
            "IMPORTANT: You MUST add one GET /ledger/account?number=X step for EACH account number "
            "mentioned in the prompt. Opening balance typically involves 2-5 accounts. Then create "
            "ONE POST /ledger/voucher/openingBalance with ALL postings.\n"
            "Do NOT fetch all accounts — only GET the specific accounts mentioned in the task.\n"
            "CRITICAL: All postings MUST sum to zero (total debit = total credit). If the task only "
            "specifies asset/liability accounts, you MUST add a balancing entry on an equity account "
            "(e.g. 2050 Annen egenkapital). GET this equity account too.\n"
            "Postings format: [{\"account\": {\"id\": <id>}, \"amountGross\": <positive_for_debit_negative_for_credit>}]"
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date", "entries"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{account_number_1}}", "fields": "id,number,name"},
                "note": "Repeat this GET for EACH account number in the task (e.g. 1920, 2400, 3000, 2050). Typically 2-5 accounts.",
            },
            {
                "method": "POST",
                "path": "/ledger/voucher/openingBalance",
                "body": {
                    "date": "{{date}}",
                    "postings": "{{postings_using_account_ids — must sum to zero}}",
                },
            },
        ],
    },

    # ===== ASSETS =====

    "create_asset": {
        "description": "Register a fixed asset (anleggsmiddel)",
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["name", "description", "acquisitionDate", "acquisitionCost", "account_number", "depreciationAccount_number"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/asset",
                "body": {
                    "name": "{{name}}",
                    "description": "{{description}}",
                    "acquisitionDate": "{{acquisitionDate}}",
                    "acquisitionCost": "{{acquisitionCost}}",
                },
            },
        ],
    },

    # ===== SALARY =====

    "create_salary_payment": {
        "description": "Create a salary transaction / salary payment for an employee",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["employee_name", "date", "year", "month", "amount", "salary_type"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
            },
            {
                "method": "GET",
                "path": "/salary/type",
                "params": {"fields": "id,number,name"},
            },
            {
                "method": "POST",
                "path": "/salary/transaction",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "date": "{{date}}",
                    "year": "{{year}}",
                    "month": "{{month}}",
                    "amount": "{{amount}}",
                    "salaryType": {"id": "$step_1.values[0].id"},
                },
            },
        ],
    },

    # ===== CUSTOMER + SUPPLIER COMBO =====

    "create_customer_supplier": {
        "description": "Create an entity that is both customer and supplier",
        "relevant_schemas": ["Customer", "Supplier"],
        "extract_fields": ["name", "email", "organizationNumber", "phoneNumber"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{name}}",
                    "isCustomer": True,
                    "isSupplier": True,
                    "email": "{{email}}",
                    "phoneNumber": "{{phoneNumber}}",
                    "organizationNumber": "{{organizationNumber}}",
                },
            },
        ],
    },

    # ===== REMINDERS =====

    "create_reminder": {
        "description": "Create a payment reminder (purring) for an overdue invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "date", "comment"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:createReminder",
                "params": {
                    "type": "SOFT_REMINDER",
                    "date": "{{date}}",
                    "comment": "{{comment}}",
                },
            },
        ],
    },

    # ===== EMPLOYEE EMPLOYMENT =====

    "create_employment": {
        "description": "Create or update employment details for an employee (ansettelsesforhold)",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["employee_name", "startDate", "employmentType", "percentageOfFullTimeEquivalent"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"firstName": "{{search_firstName}}", "lastName": "{{search_lastName}}", "fields": "id,firstName,lastName"},
            },
            {
                "method": "POST",
                "path": "/employee/employment",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "startDate": "{{startDate}}",
                    "employmentType": "{{employmentType}}",
                    "percentageOfFullTimeEquivalent": "{{percentageOfFullTimeEquivalent}}",
                },
            },
        ],
    },

    # ===== NEW: UPDATE CUSTOMER =====

    "update_customer": {
        "description": "Update an existing customer's details",
        "relevant_schemas": ["Customer"],
        "extract_fields": ["customer_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name,version"},
            },
            {
                "method": "PUT",
                "path": "/customer/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== NEW: UPDATE PROJECT =====

    "update_project": {
        "description": "Update an existing project's details",
        "relevant_schemas": ["Project"],
        "extract_fields": ["project_name", "fields_to_update"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/project",
                "params": {"name": "{{project_name}}", "fields": "id,name,version"},
            },
            {
                "method": "PUT",
                "path": "/project/$step_0.values[0].id",
                "body": "{{fields_to_update}}",
            },
        ],
    },

    # ===== NEW: INVOICE WITH PAYMENT =====

    "create_invoice_with_payment": {
        "description": "Create an invoice and immediately register a payment on it",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": [
            "customer_name", "orderLines", "invoiceDate", "invoiceDueDate",
            "paymentDate", "paymentAmount", "orderDate", "deliveryDate",
        ],
        "optimal_calls": 5,
        "steps": [
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
            },
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                },
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_1.id"},
                    "orderDate": "{{orderDate}}",
                    "deliveryDate": "{{deliveryDate}}",
                    "orderLines": "{{orderLines}}",
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_2.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "invoiceDueDate": "{{invoiceDueDate}}",
                    "sendToCustomer": False,
                },
            },
            {
                "method": "PUT",
                "path": "/invoice/$step_3.id/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_0.values[0].id",
                    "paidAmount": "{{paymentAmount}}",
                },
            },
        ],
    },

    # ===== ENABLE MODULES =====

    "enable_modules": {
        "description": "Enable accounting modules on the company (e.g., invoicing, project, travel expense, salary modules)",
        "relevant_schemas": [],
        "extract_fields": ["modules"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "PUT",
                "path": "/company/modules",
                "body": "{{modules_to_enable}}",
                "note": "Enable requested modules. Module names: ACCOUNTING, INVOICE, PROJECT, EMPLOYEE, TRAVEL_EXPENSE, SALARY, etc.",
            },
        ],
    },

    # ===== FALLBACK =====

    "unknown": {
        "description": "Task type not recognized — LLM generates plan from scratch using full API reference",
        "relevant_schemas": ["Employee", "Customer", "Product", "Order", "OrderLine", "Invoice", "TravelExpense", "Project", "Department", "Contact", "Supplier", "Voucher", "Posting"],
        "extract_fields": [],
        "optimal_calls": 0,
        "steps": [],
    },
}


# Map keywords to task types for fast classification (multilingual)
KEYWORD_HINTS: dict[str, list[str]] = {
    "create_employee": ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employe", "tilsett", "tilsatt", "opprett ansatt", "ny ansatt", "create employee", "new employee"],
    "update_employee": ["oppdater ansatt", "endre ansatt", "update employee", "endre telefon", "endre epost"],
    "create_customer": ["kunde", "customer", "cliente", "client", "Kunde", "opprett kunde", "ny kunde", "registrer kunde"],
    "update_customer": ["oppdater kunde", "endre kunde", "update customer"],
    "create_product": ["produkt", "product", "producto", "produto", "Produkt", "produit", "opprett produkt", "nytt produkt"],
    "create_invoice": ["faktura", "invoice", "factura", "fatura", "Rechnung", "facture", "opprett faktura", "ny faktura"],
    "create_invoice_with_payment": ["faktura med betaling", "invoice with payment", "faktura og betaling"],
    "register_payment": ["innbetaling", "betaling", "payment", "pago", "pagamento", "Zahlung", "paiement", "registrer betaling", "registrer innbetaling"],
    "register_payment_by_search": ["betal faktura nummer", "registrer betaling pa faktura", "payment on invoice number", "betal faktura nr", "betaling for faktura", "pay invoice number", "payment for invoice", "betaling på faktura"],
    "create_credit_note": ["kreditnota", "credit note", "nota de credito", "Gutschrift", "avoir"],
    "create_travel_expense": ["reiseregning", "travel expense", "gastos de viaje", "despesas de viagem", "Reisekosten", "note de frais", "reiserekning", "registrer reiseregning"],
    "delete_travel_expense": ["slett reiseregning", "delete travel"],
    "deliver_travel_expense": ["lever reiseregning", "deliver travel expense", "send inn reiseregning"],
    "approve_travel_expense": ["godkjenn reiseregning", "approve travel expense"],
    "create_project": ["prosjekt", "project", "proyecto", "projeto", "Projekt", "projet", "opprett prosjekt", "nytt prosjekt"],
    "create_project_existing_customer": ["prosjekt for eksisterende kunde", "project for existing customer", "prosjekt eksisterende"],
    "create_internal_project": ["internt prosjekt", "internal project", "proyecto interno", "internes Projekt"],
    "update_project": ["oppdater prosjekt", "endre prosjekt", "update project"],
    "create_department": ["avdeling", "department", "departamento", "Abteilung", "departement", "opprett avdeling", "ny avdeling"],
    "create_supplier": ["leverandør", "leverandor", "supplier", "proveedor", "fornecedor", "Lieferant", "fournisseur", "opprett leverandør", "registrer leverandør", "ny leverandør"],
    "update_supplier": ["oppdater leverandor", "oppdater leverandør", "endre leverandor", "endre leverandør", "update supplier"],
    "update_department": ["oppdater avdeling", "endre avdeling", "update department"],
    "update_product": ["oppdater produkt", "endre produkt", "update product"],
    "create_contact": ["kontaktperson", "contact person", "persona de contacto", "Kontaktperson", "kontakt"],
    "create_voucher": ["bilag", "voucher", "Beleg", "piece comptable", "postering", "bokfør", "bokfor"],
    "reverse_voucher": ["reverser", "reverse", "tilbakefor"],
    "send_invoice": ["send faktura", "send invoice"],
    "create_supplier_invoice": ["leverandorfaktura", "leverandørfaktura", "supplier invoice", "inngaende faktura", "incoming invoice", "factura proveedor", "Lieferantenrechnung"],
    "create_purchase_order": ["innkjopsordre", "purchase order", "bestilling", "orden de compra", "Bestellung"],
    "bank_reconciliation": ["bankavstemming", "bank reconciliation", "kontoutskrift", "bank statement", "conciliacion bancaria", "Bankabstimmung"],
    "create_timesheet_entry": ["timeregistrering", "timeforing", "timesheet", "timer", "hours", "horas", "Stunden", "heures"],
    "create_opening_balance": ["apningsbalanse", "opening balance", "inngaende balanse", "balance inicial", "Eroeffnungsbilanz"],
    "create_asset": ["anleggsmiddel", "eiendel", "fixed asset", "activo fijo", "Anlagevermoegen"],
    "create_salary_payment": ["lonn", "salary", "loenning", "salario", "Gehalt", "salaire"],
    "create_customer_supplier": ["kunde og leverandor", "kunde og leverandør", "customer and supplier", "both customer and supplier"],
    "create_reminder": ["purring", "reminder", "betalingspaaminnelse", "Zahlungserinnerung", "rappel"],
    "create_employment": ["ansettelse", "employment", "arbeidsforhold", "empleo", "Beschaeftigung"],
    "enable_modules": ["aktiver modul", "enable module", "aktivere", "modul", "module"],
}

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
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"firstName": "{{search_firstName}}", "lastName": "{{search_lastName}}", "fields": "id,firstName,lastName"},
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
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{name}}",
                    "isCustomer": True,
                    "email": "{{email}}",
                },
            },
        ],
    },

    # ===== PRODUCTS =====

    "create_product": {
        "description": "Create a product with price and VAT settings",
        "relevant_schemas": ["Product"],
        "extract_fields": ["name", "number", "priceExcludingVatCurrency", "priceIncludingVatCurrency", "description"],
        "steps": [
            {
                "method": "POST",
                "path": "/product",
                "body": {
                    "name": "{{name}}",
                    "priceExcludingVatCurrency": "{{price}}",
                },
            },
        ],
    },

    # ===== INVOICING =====

    "create_invoice": {
        "description": "Create an invoice: customer -> order with orderLines -> invoice. NOTE: Company must have bankAccountNumber registered. If 422 about 'bankkontonummer', the sandbox is not properly set up.",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate", "customer_email"],
        "steps": [
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
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "create_invoice_existing_customer": {
        "description": "Create invoice for an existing customer (search by name first)",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate"],
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
                    "sendToCustomer": False,
                },
            },
        ],
    },

    "register_payment": {
        "description": "Register a payment on an existing invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "amount", "paymentDate"],
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
        "description": "Register a travel expense report with travel details",
        "relevant_schemas": ["TravelExpense", "TravelDetails", "TravelExpenseCost"],
        "extract_fields": ["departureDate", "returnDate", "departureFrom", "destination", "purpose", "costs", "isDayTrip", "isForeignTravel"],
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
        ],
    },

    "delete_travel_expense": {
        "description": "Delete a travel expense report",
        "relevant_schemas": ["TravelExpense"],
        "extract_fields": ["travel_expense_id"],
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
        "description": "Create a project linked to an existing customer (search by name first)",
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["project_name", "customer_name", "startDate", "endDate", "description"],
        "optimal_calls": 2,
        "steps": [
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
                    "customer": {"id": "$step_0.values[0].id"},
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "isInternal": False,
                },
            },
        ],
    },

    "create_internal_project": {
        "description": "Create an internal project (no customer). Must set projectManager.",
        "relevant_schemas": ["Project"],
        "extract_fields": ["name", "startDate", "endDate", "description"],
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
        "steps": [
            {
                "method": "POST",
                "path": "/department",
                "body": {
                    "name": "{{name}}",
                    "departmentNumber": "{{departmentNumber}}",
                },
            },
        ],
    },

    # ===== SUPPLIERS =====

    "create_supplier": {
        "description": "Create a supplier",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["name", "organizationNumber", "email", "phoneNumber"],
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{name}}",
                    "email": "{{email}}",
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
        "extract_fields": ["firstName", "lastName", "email", "customer_name"],
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
                    "customer": {"id": "$step_0.values[0].id"},
                },
            },
        ],
    },

    # ===== LEDGER / VOUCHERS =====

    "create_voucher": {
        "description": "Create a ledger voucher with postings. IMPORTANT: account numbers (e.g. 1920) are NOT IDs — you must first GET /ledger/account?number=X to find the real account ID.",
        "relevant_schemas": ["Voucher", "Posting", "Account"],
        "extract_fields": ["date", "description", "postings_with_account_numbers"],
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{debit_account_number}}", "fields": "id,number,name"},
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
                    "postings": "{{postings_using_account_ids_from_step_0_and_1}}",
                },
            },
        ],
    },

    "reverse_voucher": {
        "description": "Reverse a voucher",
        "relevant_schemas": ["Voucher"],
        "extract_fields": ["voucher_id", "date"],
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
                            {"account": {"id": "$step_1.values[0].id"}, "amount": "{{amount}}"},
                            {"account": {"id": "$step_2.values[0].id"}, "amount": "-{{amount}}"},
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
            "1. GET /bank to find the bank account ID\n"
            "2. POST /bank/reconciliation to create the reconciliation period\n"
            "3. For each transaction: POST /bank/reconciliation/match or create vouchers for unmatched items\n"
            "4. If a bank statement (CSV/file) is attached, use POST /bank/statement/import to import it first\n"
            "5. Unmatched transactions may need manual vouchers via POST /ledger/voucher\n"
            "IMPORTANT: The accounting period must be open for the reconciliation date range. "
            "If the task specifies a closing balance, the sum of matched transactions must equal it."
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date_from", "date_to", "bank_account_number", "transactions", "closing_balance"],
        "steps": [
            {
                "method": "GET",
                "path": "/bank",
                "params": {"fields": "id,accountNumber,name"},
            },
            {
                "method": "POST",
                "path": "/bank/reconciliation",
                "body": {
                    "account": {"id": "$step_0.values[0].id"},
                    "type": "MANUAL",
                    "dateTo": "{{date_to}}",
                },
            },
        ],
    },

    # ===== TIMESHEET =====

    "create_timesheet_entry": {
        "description": "Register hours/timesheet entry for an employee on a project/activity",
        "relevant_schemas": ["Employee", "Project", "Activity"],
        "extract_fields": ["employee_name", "project_name", "activity_name", "date", "hours", "comment"],
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
            "IMPORTANT: Do NOT fetch all accounts — only GET the specific accounts mentioned in the task "
            "using GET /ledger/account?number=X for each account number. This is much more efficient.\n"
            "CRITICAL: All postings MUST sum to zero (total debit = total credit). If the task only "
            "specifies asset/liability accounts, you may need a balancing entry on an equity account (e.g. 2050).\n"
            "Postings format: [{\"account\": {\"id\": <id>}, \"amountGross\": <positive_for_debit_negative_for_credit>}]"
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date", "entries"],
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{account_number_1}}", "fields": "id,number,name"},
                "note": "Fetch ONLY the specific accounts needed — repeat this step for each account number in the task",
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
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{name}}",
                    "isCustomer": True,
                    "isSupplier": True,
                    "email": "{{email}}",
                },
            },
        ],
    },

    # ===== REMINDERS =====

    "create_reminder": {
        "description": "Create a payment reminder (purring) for an overdue invoice",
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "date", "comment"],
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
            "paymentDate", "paymentAmount",
        ],
        "optimal_calls": 5,
        "steps": [
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
                    "sendToCustomer": False,
                },
            },
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
            },
            {
                "method": "PUT",
                "path": "/invoice/$step_2.id/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_3.values[0].id",
                    "paidAmount": "{{paymentAmount}}",
                },
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
    "create_employee": ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employe"],
    "update_employee": ["oppdater ansatt", "endre ansatt", "update employee", "endre telefon", "endre epost"],
    "create_customer": ["kunde", "customer", "cliente", "client", "Kunde"],
    "update_customer": ["oppdater kunde", "endre kunde", "update customer"],
    "create_product": ["produkt", "product", "producto", "produto", "Produkt", "produit"],
    "create_invoice": ["faktura", "invoice", "factura", "fatura", "Rechnung", "facture"],
    "create_invoice_with_payment": ["faktura med betaling", "invoice with payment", "faktura og betaling"],
    "register_payment": ["innbetaling", "betaling", "payment", "pago", "pagamento", "Zahlung", "paiement"],
    "create_credit_note": ["kreditnota", "credit note", "nota de credito", "Gutschrift", "avoir"],
    "create_travel_expense": ["reiseregning", "travel expense", "gastos de viaje", "despesas de viagem", "Reisekosten", "note de frais"],
    "delete_travel_expense": ["slett reiseregning", "delete travel"],
    "deliver_travel_expense": ["lever reiseregning", "deliver travel expense", "send inn reiseregning"],
    "approve_travel_expense": ["godkjenn reiseregning", "approve travel expense"],
    "create_project": ["prosjekt", "project", "proyecto", "projeto", "Projekt", "projet"],
    "create_project_existing_customer": ["prosjekt for eksisterende kunde", "project for existing customer", "prosjekt eksisterende"],
    "create_internal_project": ["internt prosjekt", "internal project", "proyecto interno"],
    "update_project": ["oppdater prosjekt", "endre prosjekt", "update project"],
    "create_department": ["avdeling", "department", "departamento", "Abteilung", "departement"],
    "create_supplier": ["leverandor", "supplier", "proveedor", "fornecedor", "Lieferant", "fournisseur"],
    "update_supplier": ["oppdater leverandor", "endre leverandor", "update supplier"],
    "update_department": ["oppdater avdeling", "endre avdeling", "update department"],
    "update_product": ["oppdater produkt", "endre produkt", "update product"],
    "create_contact": ["kontaktperson", "contact person", "persona de contacto", "Kontaktperson"],
    "create_voucher": ["bilag", "voucher", "Beleg", "piece comptable"],
    "reverse_voucher": ["reverser", "reverse", "tilbakefor"],
    "send_invoice": ["send faktura", "send invoice"],
    "create_supplier_invoice": ["leverandorfaktura", "supplier invoice", "inngaende faktura", "incoming invoice", "factura proveedor", "Lieferantenrechnung"],
    "create_purchase_order": ["innkjopsordre", "purchase order", "bestilling", "orden de compra", "Bestellung"],
    "bank_reconciliation": ["bankavstemming", "bank reconciliation", "kontoutskrift", "bank statement", "conciliacion bancaria", "Bankabstimmung"],
    "create_timesheet_entry": ["timeregistrering", "timeforing", "timesheet", "timer", "hours", "horas", "Stunden", "heures"],
    "create_opening_balance": ["apningsbalanse", "opening balance", "inngaende balanse", "balance inicial", "Eroeffnungsbilanz"],
    "create_asset": ["anleggsmiddel", "eiendel", "fixed asset", "activo fijo", "Anlagevermoegen"],
    "create_salary_payment": ["lonn", "salary", "loenning", "salario", "Gehalt", "salaire"],
    "create_customer_supplier": ["kunde og leverandor", "customer and supplier", "both customer and supplier"],
    "create_reminder": ["purring", "reminder", "betalingspaaminnelse", "Zahlungserinnerung", "rappel"],
    "create_employment": ["ansettelse", "employment", "arbeidsforhold", "empleo", "Beschaeftigung"],
}

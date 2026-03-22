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
        "extract_fields": ["firstName", "lastName", "email", "dateOfBirth", "phoneNumberMobile", "role", "employeeNumber", "addressLine1", "postalCode", "city", "startDate"],
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
                "depends_on": [0],
                "body": {
                    "firstName": "{{firstName}}",
                    "lastName": "{{lastName}}",
                    "email": "{{email}}",
                    "dateOfBirth": "{{dateOfBirth}}",
                    "phoneNumberMobile": "{{phoneNumberMobile}}",
                    "employeeNumber": "{{employeeNumber}}",
                    "userType": "STANDARD",
                    "department": {"id": "$step_0.values[0].id"},
                    "address": {
                        "addressLine1": "{{addressLine1}}",
                        "postalCode": "{{postalCode}}",
                        "city": "{{city}}",
                    },
                },
            },
        ],
        "conditional_steps": {
            "if_role": {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {"employeeId": "$step_1.id", "template": "{{role}}"},
            },
            "if_startDate": {
                "method": "POST",
                "path": "/employee/employment",
                "body": {"employee": {"id": "$step_1.id"}, "startDate": "{{startDate}}"},
            },
        },
    },

    "update_employee": {
        "description": (
            "Update an existing employee's details (phone, email, address, etc.).\n"
            "IMPORTANT: PUT body MUST include 'id' and 'version' from the GET response.\n"
            "The fields_to_update should be merged with id+version in the PUT body.\n"
            "Example: if updating email, PUT body = {\"id\": X, \"version\": Y, \"email\": \"new@email.com\"}"
        ),
        "relevant_schemas": ["Employee"],
        "extract_fields": ["search_firstName", "search_lastName", "fields_to_update"],
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
                "note": "MUST include id and version from GET. Template engine injects these automatically.",
            },
        ],
    },

    # ===== CUSTOMERS =====

    "create_customer": {
        "description": "Create a customer with contact details and optional address",
        "relevant_schemas": ["Customer"],
        "extract_fields": ["name", "email", "organizationNumber", "phoneNumber", "phoneNumberMobile", "description", "website", "isPrivateIndividual", "isSupplier", "addressLine1", "postalCode", "city"],
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
                    "phoneNumberMobile": "{{phoneNumberMobile}}",
                    "description": "{{description}}",
                    "website": "{{website}}",
                    "isPrivateIndividual": "{{isPrivateIndividual}}",
                    "organizationNumber": "{{organizationNumber}}",
                    "postalAddress": {
                        "addressLine1": "{{addressLine1}}",
                        "postalCode": "{{postalCode}}",
                        "city": "{{city}}",
                    },
                },
            },
        ],
    },

    # ===== PRODUCTS =====

    "create_product": {
        "description": "Create a product with price and VAT settings",
        "relevant_schemas": ["Product"],
        "extract_fields": ["name", "number", "priceExcludingVatCurrency", "priceIncludingVatCurrency", "description", "vatTypeId"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/product",
                "body": {
                    "name": "{{name}}",
                    "number": "{{number}}",
                    "priceExcludingVatCurrency": "{{priceExcludingVatCurrency}}",
                    "priceIncludingVatCurrency": "{{priceIncludingVatCurrency}}",
                    "description": "{{description}}",
                    "vatType": {"id": "{{vatTypeId}}"},
                },
            },
        ],
    },

    # ===== INVOICING =====

    "create_invoice": {
        "description": "Create an invoice: POST customer -> POST order (with orderLines in body) -> PUT order/:invoice",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "customer_email", "customer_organizationNumber",
                          "customer_phoneNumber", "customer_phoneNumberMobile",
                          "customer_addressLine1", "customer_postalCode", "customer_city",
                          "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "phoneNumberMobile": "{{customer_phoneNumberMobile}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
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
                "note": "orderLines go IN the order body as array: [{description, count, unitPriceExcludingVatCurrency}]",
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
        "description": "Create invoice for an existing customer: GET customer -> POST order (with orderLines) -> PUT order/:invoice",
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
                "note": "orderLines go IN the order body as array: [{description, count, unitPriceExcludingVatCurrency}]",
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
                "params": {"invoiceNumber": "{{invoiceNumber}}", "invoiceDateFrom": "2020-01-01", "invoiceDateTo": "2030-12-31", "fields": "id,invoiceNumber,amount"},
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
        "extract_fields": ["departureDate", "returnDate", "departureFrom", "destination", "purpose", "costs", "isDayTrip", "isForeignTravel", "title", "cost_amount", "cost_description_if_any", "perDiem_dailyRate", "perDiem_days", "employeeEmail", "employeeFirstName", "employeeLastName"],
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
                    "firstName": "{{employeeFirstName}}",
                    "lastName": "{{employeeLastName}}",
                    "email": "{{employeeEmail}}",
                    "userType": "STANDARD",
                    "department": {"id": "$step_0.values[0].id"},
                },
                "note": "Create employee (sandbox is empty). If firstName/lastName not extracted, will be cleaned up.",
            },
            {
                "method": "POST",
                "path": "/travelExpense",
                "body": {
                    "employee": {"id": "$step_1.id"},
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
        "conditional_steps": {
            "if_cost_amount": [
                {
                    "method": "GET",
                    "path": "/travelExpense/paymentType",
                    "params": {"fields": "id,description"},
                },
                {
                    "method": "POST",
                    "path": "/travelExpense/cost",
                    "body": {
                        "travelExpense": {"id": "$step_2.id"},
                        "vatType": {"id": 0},
                        "paymentType": {"id": "$step_3.values[0].id"},
                        "amountCurrencyIncVat": "{{cost_amount}}",
                        "date": "{{departureDate}}",
                        "comments": "{{cost_description_if_any}}",
                    },
                    "note": "vatType 0 = exempt (safe default, avoids VAT_NOT_REGISTERED). currency omitted (defaults to NOK).",
                },
            ],
        },
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
        "description": (
            "Create a project linked to a NEW customer. Must set projectManager with proper entitlements.\n"
            "If the prompt names a specific person as project manager (e.g. 'Sofia Costa'), create them as an employee first "
            "(POST /employee), then grant ALL_PRIVILEGES entitlement, then create the project with that employee as projectManager.\n"
            "Steps 1 (POST /customer) and 2 (PUT entitlement) can run in parallel since they have no dependency on each other."
        ),
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["project_name", "customer_name", "customer_email", "customer_organizationNumber", "customer_phoneNumber", "customer_phoneNumberMobile", "customer_description", "customer_website", "customer_isPrivateIndividual", "customer_addressLine1", "customer_postalCode", "customer_city", "startDate", "endDate", "isInternal", "projectManager", "project_description"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
                "note": "Get default employee to use as project manager. If prompt names a specific person, use POST /employee to create them instead.",
            },
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "phoneNumberMobile": "{{customer_phoneNumberMobile}}",
                    "description": "{{customer_description}}",
                    "website": "{{customer_website}}",
                    "isPrivateIndividual": "{{customer_isPrivateIndividual}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
                },
                "note": "Can run in parallel with step 2 (entitlement grant).",
            },
            {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {
                    "employeeId": "$step_0.values[0].id",
                    "template": "ALL_PRIVILEGES",
                },
                "note": "Grant project manager entitlements. Can run in parallel with step 1 (customer creation). Must complete before step 3.",
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
                "note": "Depends on step 1 (customer) and step 2 (entitlement).",
            },
        ],
    },

    "create_project_existing_customer": {
        "description": (
            "Create a project linked to an existing customer (search by name first). Must set projectManager with proper entitlements.\n"
            "If the prompt names a specific person as project manager, create them as an employee first "
            "(POST /employee), then grant ALL_PRIVILEGES entitlement, then create the project.\n"
            "Steps 0 (GET /employee) and 1 (GET /customer) can run in parallel. Step 2 depends on step 0. Step 3 depends on steps 1+2."
        ),
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["project_name", "customer_name", "startDate", "endDate", "project_description", "projectManager"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
                "note": "Can run in parallel with step 1 (customer search). If prompt names a specific person, use POST /employee instead.",
            },
            {
                "method": "GET",
                "path": "/customer",
                "params": {"name": "{{customer_name}}", "fields": "id,name"},
                "note": "Can run in parallel with step 0 (employee lookup).",
            },
            {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {
                    "employeeId": "$step_0.values[0].id",
                    "template": "ALL_PRIVILEGES",
                },
                "note": "Grant project manager entitlements. Depends on step 0. Must complete before step 3.",
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
                "note": "Depends on step 1 (customer) and step 2 (entitlement).",
            },
        ],
    },

    "create_internal_project": {
        "description": (
            "Create an internal project (no customer). Must set projectManager with proper entitlements.\n"
            "If the prompt names a specific person as project manager, create them as an employee first "
            "(POST /employee), then grant ALL_PRIVILEGES entitlement, then create the project."
        ),
        "relevant_schemas": ["Project"],
        "extract_fields": ["project_name", "startDate", "endDate", "project_description"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
                "note": "Get default employee for project manager. If prompt names a specific person, use POST /employee instead.",
            },
            {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {
                    "employeeId": "$step_0.values[0].id",
                    "template": "ALL_PRIVILEGES",
                },
                "note": "Grant project manager entitlements. Depends on step 0. Must complete before step 2.",
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
                "note": "Depends on step 1 (entitlement).",
            },
        ],
    },

    # ===== DEPARTMENTS =====

    "create_department": {
        "description": "Create a department",
        "relevant_schemas": ["Department"],
        "extract_fields": ["name", "departmentNumber", "departmentManagerId"],
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
        "description": "Create a supplier with optional address",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["name", "organizationNumber", "email", "phoneNumber", "phoneNumberMobile", "description", "addressLine1", "postalCode", "city"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{name}}",
                    "email": "{{email}}",
                    "phoneNumber": "{{phoneNumber}}",
                    "phoneNumberMobile": "{{phoneNumberMobile}}",
                    "description": "{{description}}",
                    "organizationNumber": "{{organizationNumber}}",
                    "postalAddress": {
                        "addressLine1": "{{addressLine1}}",
                        "postalCode": "{{postalCode}}",
                        "city": "{{city}}",
                    },
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
                    "phoneNumberMobile": "{{phoneNumber}}",
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
            "POSTING FORMAT: Each posting MUST have: row (starting from 1, NEVER 0), account.id, "
            "amountGross, amountGrossCurrency (same as amountGross), and vatType.id.\n"
            "VATTYPE IDs (standard, hardcode these — same in ALL Tripletex sandboxes):\n"
            "  vatType 0 = No VAT — for bank/asset accounts (1xxx, 2xxx)\n"
            "  vatType 3 = Outgoing VAT 25% — for revenue accounts (3xxx)\n"
            "  vatType 1 = Incoming VAT 25% — for expense accounts (6xxx, 7xxx)\n"
            "ALWAYS include vatType in EVERY posting. Use 0 if unsure."
        ),
        "relevant_schemas": ["Voucher", "Posting", "Account"],
        "extract_fields": ["date", "description", "postings_with_account_numbers", "debit_account_number", "credit_account_number", "debit_amount", "credit_amount"],
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
                        {"row": 1, "account": {"id": "$step_0.values[0].id"}, "amountGross": "{{debit_amount}}", "amountGrossCurrency": "{{debit_amount}}", "vatType": {"id": 0}},
                        {"row": 2, "account": {"id": "$step_1.values[0].id"}, "amountGross": "-{{credit_amount}}", "amountGrossCurrency": "-{{credit_amount}}", "vatType": {"id": 3}},
                    ],
                },
                "note": "Row starts from 1 (NEVER 0). vatType: 0=no VAT (1xxx,2xxx), 3=outgoing 25% (3xxx), 1=incoming 25% (6xxx,7xxx). ALWAYS include vatType.",
            },
        ],
    },

    "reverse_voucher": {
        "description": (
            "Reverse a voucher. If voucher_id is known, reverse directly. "
            "If only a description is given, search for it first via GET /ledger/voucher."
        ),
        "relevant_schemas": ["Voucher"],
        "extract_fields": ["voucher_id", "voucher_description", "date", "dateFrom", "dateTo"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/ledger/voucher",
                "params": {
                    "dateFrom": "{{dateFrom}}",
                    "dateTo": "{{dateTo}}",
                    "fields": "id,date,description,number",
                },
                "note": "Search for the voucher. dateFrom/dateTo are required params. Use dates from prompt context.",
            },
            {
                "method": "PUT",
                "path": "/ledger/voucher/$step_0.values[-1].id/:reverse",
                "params": {"date": "{{date}}"},
                "note": "Reverse the found voucher. If voucher_id was extracted, template_engine overrides the path.",
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
        "description": (
            "Create a supplier invoice (incoming invoice from a supplier).\n"
            "Steps: 1) Create supplier, 2) GET expense account, 3) GET AP account (2400), 4) POST /supplierInvoice.\n"
            "CRITICAL: Must use POST /supplierInvoice (NOT /ledger/voucher) — scoring checks SupplierInvoice entity.\n"
            "Required fields: invoiceNumber, invoiceDate, invoiceDueDate, amountCurrency, supplier.id, voucher with postings."
        ),
        "relevant_schemas": ["Supplier", "SupplierInvoice", "Voucher", "Posting"],
        "extract_fields": ["supplier_name", "supplier_organizationNumber", "supplier_email", "supplier_phoneNumber", "supplier_phoneNumberMobile", "supplier_description", "supplier_addressLine1", "supplier_postalCode", "supplier_city", "invoiceNumber", "invoiceDate", "dueDate", "amount", "account_number", "description", "expense_account_number"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{supplier_name}}",
                    "organizationNumber": "{{supplier_organizationNumber}}",
                    "email": "{{supplier_email}}",
                    "phoneNumber": "{{supplier_phoneNumber}}",
                    "phoneNumberMobile": "{{supplier_phoneNumberMobile}}",
                    "description": "{{supplier_description}}",
                    "postalAddress": {
                        "addressLine1": "{{supplier_addressLine1}}",
                        "postalCode": "{{supplier_postalCode}}",
                        "city": "{{supplier_city}}",
                    },
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
                    "invoiceDueDate": "{{dueDate}}",
                    "amountCurrency": "{{amount}}",
                    "supplier": {"id": "$step_0.id"},
                    "voucher": {
                        "date": "{{invoiceDate}}",
                        "description": "Leverandørfaktura {{invoiceNumber}} fra {{supplier_name}}",
                        "postings": [
                            {"row": 1, "account": {"id": "$step_1.values[0].id"}, "amountGross": "{{amount}}", "amountGrossCurrency": "{{amount}}", "vatType": {"id": 1}, "supplier": {"id": "$step_0.id"}},
                            {"row": 2, "account": {"id": "$step_2.values[0].id"}, "amountGross": "-{{amount}}", "amountGrossCurrency": "-{{amount}}", "vatType": {"id": 0}, "supplier": {"id": "$step_0.id"}},
                        ],
                    },
                },
                "fallback_on_500": {
                    "path": "/ledger/voucher",
                    "body": {
                        "date": "{{invoiceDate}}",
                        "description": "Leverandørfaktura {{invoiceNumber}} fra {{supplier_name}}",
                        "postings": [
                            {"row": 1, "account": {"id": "$step_1.values[0].id"}, "amountGross": "{{amount}}", "amountGrossCurrency": "{{amount}}", "vatType": {"id": 1}, "supplier": {"id": "$step_0.id"}},
                            {"row": 2, "account": {"id": "$step_2.values[0].id"}, "amountGross": "-{{amount}}", "amountGrossCurrency": "-{{amount}}", "vatType": {"id": 0}, "supplier": {"id": "$step_0.id"}},
                        ],
                    },
                },
                "note": "Try /supplierInvoice first, fall back to /ledger/voucher on 500.",
            },
        ],
    },

    # ===== PURCHASE ORDERS =====

    "create_purchase_order": {
        "description": (
            "Create a purchase order to a supplier. IMPORTANT:\n"
            "- ourContact (employee ref) is REQUIRED on the purchase order.\n"
            "- orderLines CANNOT be included in the POST /purchaseOrder body (they need a purchaseOrder reference).\n"
            "- Solution: Create the purchase order first WITHOUT orderLines, then POST each orderLine separately via POST /purchaseOrder/orderline.\n"
            "VALID POST /purchaseOrder body fields: supplier.id (required), ourContact.id (required), deliveryDate (required). NOTHING ELSE.\n"
            "FORBIDDEN FIELDS that cause 422 'Oppdatering av dette feltet er ikke tillatt':\n"
            "  - status (read-only)\n"
            "  - currency (read-only)\n"
            "  - transportType (read-only)\n"
            "  - orderLineSorting (read-only)\n"
            "  - orderLines (must be added via separate POST /purchaseOrder/orderline)\n"
            "  - receiverEmail, orderDate, reference, comment, number (all forbidden)\n"
            "- NOTE: POST /purchaseOrder requires moduleOrderOut to be enabled. If you get 'Oppdatering av dette feltet er ikke tillatt' on ALL fields, the module is not active.\n"
            "  Competition sandboxes have this module enabled."
        ),
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["supplier_name", "supplier_organizationNumber", "supplier_email", "supplier_phoneNumber", "supplier_phoneNumberMobile", "supplier_description", "supplier_addressLine1", "supplier_postalCode", "supplier_city", "deliveryDate", "orderLine_description", "orderLine_count", "orderLine_unitPriceExcludingVatCurrency", "orderLines"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id", "count": 1},
                "note": "Step 0: Get employee for ourContact (REQUIRED on purchase order).",
            },
            {
                "method": "POST",
                "path": "/supplier",
                "body": {
                    "name": "{{supplier_name}}",
                    "organizationNumber": "{{supplier_organizationNumber}}",
                    "email": "{{supplier_email}}",
                    "phoneNumber": "{{supplier_phoneNumber}}",
                    "phoneNumberMobile": "{{supplier_phoneNumberMobile}}",
                    "description": "{{supplier_description}}",
                    "postalAddress": {
                        "addressLine1": "{{supplier_addressLine1}}",
                        "postalCode": "{{supplier_postalCode}}",
                        "city": "{{supplier_city}}",
                    },
                },
                "note": "Step 1: Create supplier.",
            },
            {
                "method": "POST",
                "path": "/purchaseOrder",
                "body": {
                    "supplier": {"id": "$step_1.id"},
                    "ourContact": {"id": "$step_0.values[0].id"},
                    "deliveryDate": "{{deliveryDate}}",
                },
                "note": "Step 2: Create purchase order WITHOUT orderLines (they must be added separately).",
            },
            {
                "method": "POST",
                "path": "/purchaseOrder/orderline",
                "body": {
                    "purchaseOrder": {"id": "$step_2.id"},
                    "description": "{{orderLine_description}}",
                    "count": "{{orderLine_count}}",
                    "unitPriceExcludingVatCurrency": "{{orderLine_unitPriceExcludingVatCurrency}}",
                },
                "note": "Step 3: Add order line to the purchase order. Repeat for each line.",
            },
        ],
    },

    # ===== BANK RECONCILIATION =====

    "bank_reconciliation": {
        "description": (
            "Create a bank reconciliation. Multi-step process:\n"
            "1. GET /bank to find bank account (match by accountNumber if specified)\n"
            "2. POST /bank/reconciliation with account.id, type=MANUAL, dateFrom, dateTo\n"
            "3. If CSV/file attached: parse transactions and create vouchers via POST /ledger/voucher\n"
            "4. For each unmatched transaction: POST /bank/reconciliation/match\n"
            "IMPORTANT: dateFrom and dateTo MUST be within an open accounting period.\n"
            "closingBalance should match the sum of all transactions if specified.\n"
            "GET /ledger/account for each account number before creating vouchers."
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date_from", "date_to", "bank_account_number", "transactions", "closing_balance"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/bank",
                "params": {"fields": "id,accountNumber,name"},
                "note": "Find the bank account. Match by accountNumber if specified in the task.",
            },
            {
                "method": "GET",
                "path": "/ledger/accountingPeriod",
                "params": {"fields": "id,start,end,isClosed", "count": "1", "periodEnd": "{{date_to}}"},
                "note": "Find the accounting period that covers the date range. Use periodEnd to filter.",
            },
            {
                "method": "POST",
                "path": "/bank/reconciliation",
                "body": {
                    "account": {"id": "$step_0.values[0].id"},
                    "type": "MANUAL",
                    "dateFrom": "{{date_from}}",
                    "accountingPeriod": {"id": "$step_1.values[0].id"},
                },
                "note": "dateFrom is required. Use accountingPeriod instead of dateTo (dateTo does NOT exist).",
            },
        ],
    },

    # ===== TIMESHEET =====

    "create_timesheet_entry": {
        "description": (
            "Register hours/timesheet entry for an employee on a project/activity.\n"
            "IMPORTANT: The timesheet date MUST be on or after the project's startDate. "
            "If the prompt doesn't specify a date, use today's date but ensure it falls within the project's date range.\n"
            "Use an activity where isProjectActivity=true (e.g. 'Fakturerbart arbeid' or 'Prosjektadministrasjon').\n"
            "VALID POST /timesheet/entry body fields: employee, project, activity, date, hours, comment.\n"
            "FORBIDDEN fields that cause 422: description, title, name, type. Use 'comment' for any text, NOT 'description'."
        ),
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
                "params": {"name": "{{project_name}}", "fields": "id,name,startDate,endDate,version"},
            },
            {
                "method": "GET",
                "path": "/activity",
                "params": {"fields": "id,name,isProjectActivity"},
                "note": "Pick an activity where isProjectActivity=true",
            },
            {
                "method": "PUT",
                "path": "/project/$step_1.values[0].id",
                "body": {
                    "id": "$step_1.values[0].id",
                    "version": "$step_1.values[0].version",
                    "startDate": "{{date}}",
                },
                "note": "Ensure project startDate <= timesheet date. If already OK, this is a no-op update.",
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
                "note": "Ensure date >= project startDate from step 1 response. If date is before project startDate, use project startDate instead.",
            },
        ],
    },

    # ===== OPENING BALANCE =====

    "create_opening_balance": {
        "description": (
            "Set opening balance entries. CRITICAL RULES:\n"
            "1. Add ONE GET /ledger/account?number=X step for EACH account number in the prompt\n"
            "2. ALL postings MUST sum to zero (debit = credit)\n"
            "3. If only asset accounts given, add balancing equity posting (account 2050)\n"
            "4. GET the equity account too if you need to add a balancing entry\n"
            "5. Each posting: {row: N, account: {id: X}, amountGross: Y, amountGrossCurrency: Y}\n"
            "6. Positive = debit, negative = credit\n"
            "7. Row numbers start from 1 (NEVER 0)"
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["date", "entries", "account_number_1"],
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
                "path": "/ledger/voucher",
                "body": {
                    "date": "{{date}}",
                    "description": "Åpningsbalanse",
                    "postings": "{{postings_using_account_ids — must sum to zero}}",
                },
                "note": "Use regular /ledger/voucher for opening balance. Row starts from 1. Include vatType on each posting (0 for 1xxx/2xxx accounts).",
            },
        ],
    },

    # ===== ASSETS =====

    "create_asset": {
        "description": "Register a fixed asset (anleggsmiddel). NOTE: The date field is 'dateOfAcquisition' (NOT 'acquisitionDate'). Requires moduleFixedAssetRegister to be enabled. If POST /asset returns permission error, the module may need to be enabled first via the Tripletex web UI or via PUT /company/modules.",
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": ["name", "description", "dateOfAcquisition", "acquisitionCost", "account_number", "depreciationAccount_number"],
        "optimal_calls": 1,
        "steps": [
            {
                "method": "POST",
                "path": "/asset",
                "body": {
                    "name": "{{name}}",
                    "description": "{{description}}",
                    "dateOfAcquisition": "{{dateOfAcquisition}}",
                    "acquisitionCost": "{{acquisitionCost}}",
                },
            },
        ],
    },

    # ===== SALARY =====

    "create_salary_payment": {
        "description": (
            "Create a salary transaction for an employee.\n"
            "NOTE: POST /salary/transaction returns 403 'You do not have permission' in dev sandbox\n"
            "because the salary module is not enabled. Competition sandboxes have this module enabled.\n"
            "POST /salary/transaction body MUST have: year (int), month (int), payslips (array).\n"
            "Each payslip has: employee: {id: X}.\n"
            "FORBIDDEN fields that cause 422: employeeId, amount, rate, count, salaryType, salaryTypeId.\n"
            "The salary/transaction endpoint creates a payslip for the given month/year.\n"
            "Field names for salary/transaction may vary. The LLM should adapt based on API error messages."
        ),
        "relevant_schemas": ["Employee"],
        "extract_fields": ["employee_name", "year", "month"],
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
                "params": {"fields": "id,number,name", "count": 10},
                "note": "Fetch salary types to find valid type IDs for transaction.",
            },
            {
                "method": "POST",
                "path": "/salary/transaction",
                "body": {
                    "year": "{{year}}",
                    "month": "{{month}}",
                    "payslips": [{"employee": {"id": "$step_0.values[0].id"}}],
                },
                "note": "Body is ONLY: year, month, payslips. NO other fields. payslips is array of {employee: {id}}. Requires salary module permission.",
            },
        ],
    },

    # ===== CUSTOMER + SUPPLIER COMBO =====

    "create_customer_supplier": {
        "description": "Create an entity that is both customer and supplier, with optional address",
        "relevant_schemas": ["Customer", "Supplier"],
        "extract_fields": ["name", "email", "organizationNumber", "phoneNumber", "phoneNumberMobile", "description", "website", "isPrivateIndividual", "addressLine1", "postalCode", "city"],
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
                    "phoneNumberMobile": "{{phoneNumberMobile}}",
                    "description": "{{description}}",
                    "website": "{{website}}",
                    "isPrivateIndividual": "{{isPrivateIndividual}}",
                    "organizationNumber": "{{organizationNumber}}",
                    "postalAddress": {
                        "addressLine1": "{{addressLine1}}",
                        "postalCode": "{{postalCode}}",
                        "city": "{{city}}",
                    },
                },
            },
        ],
    },

    # ===== REMINDERS =====

    "create_reminder": {
        "description": (
            "Create a payment reminder (purring) for an overdue invoice.\n"
            "STEP 1: GET the invoice to check amountOutstanding > 0 (cannot remind on paid invoices).\n"
            "STEP 2: PUT /:createReminder with REQUIRED query params: type, date, dispatchType.\n"
            "CRITICAL: The param name is 'dispatchType' — NOT 'sendMethod', NOT 'sendType', NOT 'sendTypes'.\n"
            "Using wrong param names (sendType etc.) causes 'Minst en sendetype ma oppgis' error.\n"
            "Valid dispatchTypes: EMAIL, SMS, LETTER.\n"
            "Valid types: SOFT_REMINDER, REMINDER, NOTICE_OF_DEBT_COLLECTION.\n"
            "NOTE: date must be >= invoice due date, otherwise you get 422.\n"
            "NOTE: Cannot create reminder for a paid invoice (amountOutstanding == 0).\n"
            "All params go in query string, NOT in the request body."
        ),
        "relevant_schemas": ["Invoice"],
        "extract_fields": ["invoice_id", "date", "comment"],
        "optimal_calls": 2,
        "steps": [
            {
                "method": "GET",
                "path": "/invoice/{{invoice_id}}",
                "params": {"fields": "id,amountOutstanding,invoiceNumber,invoiceDueDate"},
                "note": "Check invoice is unpaid (amountOutstanding > 0) before creating reminder.",
            },
            {
                "method": "PUT",
                "path": "/invoice/{{invoice_id}}/:createReminder",
                "params": {
                    "type": "SOFT_REMINDER",
                    "date": "{{date}}",
                    "dispatchType": "EMAIL",
                },
                "note": "dispatchType is REQUIRED (NOT sendMethod/sendType). Use EMAIL as default. Date must be >= invoice dueDate.",
            },
        ],
    },

    # ===== EMPLOYEE EMPLOYMENT =====

    "create_employment": {
        "description": (
            "Create employment + employment details for an employee (ansettelsesforhold).\n"
            "Step 1: GET employee. Step 2: PUT employee (set dateOfBirth). "
            "Step 3: POST /employee/employment (creates employment record). "
            "Step 4: POST /employee/employment/details (sets percentage, salary, occupation code etc.).\n"
            "Employment and employment/details are SEPARATE endpoints!\n"
            "Employee MUST have dateOfBirth set before employment can be created."
        ),
        "relevant_schemas": ["Employee"],
        "extract_fields": ["search_firstName", "search_lastName", "startDate", "dateOfBirth", "percentageOfFullTimeEquivalent", "annualSalary", "occupationCode"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"firstName": "{{search_firstName}}", "lastName": "{{search_lastName}}", "fields": "id,firstName,lastName,dateOfBirth,version"},
            },
            {
                "method": "PUT",
                "path": "/employee/$step_0.values[0].id",
                "body": {
                    "id": "$step_0.values[0].id",
                    "version": "$step_0.values[0].version",
                    "dateOfBirth": "{{dateOfBirth}}",
                },
                "note": "Set dateOfBirth if missing. Use date from prompt or default 1990-01-01.",
            },
            {
                "method": "POST",
                "path": "/employee/employment",
                "body": {
                    "employee": {"id": "$step_0.values[0].id"},
                    "startDate": "{{startDate}}",
                },
                "note": "Creates the employment record. ONLY employee.id and startDate are valid here.",
            },
            {
                "method": "POST",
                "path": "/employee/employment/details",
                "body": {
                    "employment": {"id": "$step_2.id"},
                    "date": "{{startDate}}",
                    "percentageOfFullTimeEquivalent": "{{percentageOfFullTimeEquivalent}}",
                    "annualSalary": "{{annualSalary}}",
                    "occupationCode": "{{occupationCode}}",
                },
                "note": "Sets employment details like percentage, salary, occupation code. Separate endpoint from /employee/employment.",
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
        "description": (
            "Create an invoice and immediately register full payment on it.\n"
            "CRITICAL: The POST /order body MUST include 'orderLines' as an array. An order without orderLines\n"
            "produces a 0 kr invoice and payment will fail. Always include at least one orderLine:\n"
            "[{\"description\": \"...\", \"count\": N, \"unitPriceExcludingVatCurrency\": X}]\n"
            "CRITICAL: paidAmount MUST equal the total invoice amount (quantity x unit price for all lines). "
            "If the prompt says '1 stk a 5000 kr', paidAmount = 5000. For '2 stk a 3000 kr', paidAmount = 6000.\n"
            "The /:invoice action returns the invoice. Use $step_3.id for the /:payment path."
        ),
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": [
            "customer_name", "customer_email", "customer_organizationNumber",
            "customer_phoneNumber", "customer_phoneNumberMobile",
            "customer_addressLine1", "customer_postalCode", "customer_city",
            "orderLines", "invoiceDate", "invoiceDueDate",
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
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "phoneNumberMobile": "{{customer_phoneNumberMobile}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
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
                "note": "MANDATORY: orderLines MUST be included in this body. Without orderLines the invoice will be 0 kr. Format: [{\"description\": \"...\", \"count\": N, \"unitPriceExcludingVatCurrency\": X}]. NEVER omit orderLines.",
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
                "note": "paidAmount MUST equal total invoice amount. Calculate: sum of (count × unitPrice) for all order lines.",
            },
        ],
    },

    # ===== ENABLE MODULES =====

    "enable_modules": {
        "description": "Enable accounting modules on the company (e.g., invoicing, project, travel expense, salary modules)",
        "relevant_schemas": [],
        "extract_fields": ["modules_to_enable"],
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

    # ===== INVOICE + SEND =====

    "create_invoice_and_send": {
        "description": "Create an invoice AND send it to the customer via email. Steps: POST customer -> POST order (with orderLines) -> PUT order/:invoice -> PUT invoice/:send",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "customer_email", "customer_organizationNumber",
                          "customer_phoneNumber", "customer_phoneNumberMobile",
                          "customer_addressLine1", "customer_postalCode", "customer_city",
                          "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate", "amount"],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "phoneNumberMobile": "{{customer_phoneNumberMobile}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
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
                "note": "orderLines go IN the order body as array: [{description, count, unitPriceExcludingVatCurrency}]",
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
            {
                "method": "PUT",
                "path": "/invoice/$step_2.id/:send",
                "params": {
                    "sendType": "EMAIL",
                },
            },
        ],
    },

    # ===== FIXED PRICE PROJECT + INVOICE =====

    "fixed_price_project_invoice": {
        "description": (
            "Create a fixed-price project and invoice the customer for a percentage of the fixed price.\n"
            "Steps: POST customer -> GET department -> POST employee (project manager) -> PUT entitlement -> "
            "POST project(isFixedPrice=true, fixedprice=X) -> POST order(amount=fixedprice*pct/100) -> PUT order/:invoice"
        ),
        "relevant_schemas": ["Customer", "Project", "Employee", "Order", "OrderLine", "Invoice"],
        "extract_fields": [
            "customer_name", "customer_organizationNumber", "customer_email",
            "customer_phoneNumber", "customer_addressLine1", "customer_postalCode", "customer_city",
            "project_name", "fixedprice", "invoice_percentage",
            "projectManager_firstName", "projectManager_lastName", "projectManager_email",
        ],
        "optimal_calls": 7,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "email": "{{customer_email}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
                },
            },
            {
                "method": "GET",
                "path": "/department",
                "params": {"fields": "id,name", "count": 1},
            },
            {
                "method": "POST",
                "path": "/employee",
                "depends_on": [1],
                "body": {
                    "firstName": "{{projectManager_firstName}}",
                    "lastName": "{{projectManager_lastName}}",
                    "email": "{{projectManager_email}}",
                    "userType": "STANDARD",
                    "department": {"id": "$step_1.values[0].id"},
                },
            },
            {
                "method": "PUT",
                "path": "/employee/entitlement/:grantEntitlementsByTemplate",
                "params": {"employeeId": "$step_2.id", "template": "ALL_PRIVILEGES"},
            },
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "isFixedPrice": True,
                    "fixedprice": "{{fixedprice}}",
                    "projectManager": {"id": "$step_2.id"},
                    "department": {"id": "$step_1.values[0].id"},
                    "customer": {"id": "$step_0.id"},
                    "startDate": "{{today}}",
                },
            },
            {
                "method": "POST",
                "path": "/order",
                "body": {
                    "customer": {"id": "$step_0.id"},
                    "orderDate": "{{today}}",
                    "deliveryDate": "{{today}}",
                    "orderLines": [
                        {
                            "description": "{{project_name}}",
                            "count": 1,
                            "unitPriceExcludingVatCurrency": "{{invoice_amount}}",
                        },
                    ],
                },
            },
            {
                "method": "PUT",
                "path": "/order/$step_5.id/:invoice",
                "params": {
                    "invoiceDate": "{{today}}",
                    "invoiceDueDate": "{{invoiceDueDate}}",
                    "sendToCustomer": False,
                },
            },
        ],
    },

    # ===== DIMENSIONS + VOUCHER =====

    "create_dimensions_voucher": {
        "description": (
            "Create custom accounting dimensions with values, then register a voucher posting.\n"
            "NOTE: This template requires dynamic step expansion via _expand_dimension_steps() "
            "because the number of dimension values varies per task.\n"
            "Steps: POST /ledger/accountingDimensionName -> N x POST /ledger/accountingDimensionValue -> "
            "GET /ledger/account -> POST /ledger/voucher"
        ),
        "relevant_schemas": ["Voucher", "Posting"],
        "extract_fields": [
            "dimension_name", "dimension_values", "account_number",
            "amount", "date", "description",
        ],
        "optimal_calls": 5,
        "steps": [
            {
                "method": "POST",
                "path": "/ledger/accountingDimensionName",
                "body": {
                    "dimensionName": "{{dimension_name}}",
                },
            },
            {
                "note": "Dimension value steps are dynamically expanded by _expand_dimension_steps()",
                "method": "POST",
                "path": "/ledger/accountingDimensionValue",
                "body": {
                    "displayName": "{{first_dimension_value}}",
                    "dimensionIndex": 1,
                },
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "{{account_number}}", "fields": "id,number,name"},
            },
            {
                "method": "GET",
                "path": "/ledger/account",
                "params": {"number": "1920", "fields": "id,number,name"},
                "note": "Bank account for credit side of voucher posting",
            },
            {
                "method": "POST",
                "path": "/ledger/voucher",
                "body": {
                    "date": "{{date}}",
                    "description": "{{description}}",
                    "postings": [
                        {
                            "row": 1,
                            "account": {"id": "$step_2.values[0].id"},
                            "amountGross": "{{amount}}",
                            "amountGrossCurrency": "{{amount}}",
                            "vatType": {"id": 0},
                            "freeAccountingDimension1": {"id": "$step_1.id"},
                        },
                        {
                            "row": 2,
                            "account": {"id": "$step_3.values[0].id"},
                            "amountGross": "{{neg_amount}}",
                            "amountGrossCurrency": "{{neg_amount}}",
                            "vatType": {"id": 0},
                        },
                    ],
                },
            },
        ],
    },

    # ===== FULL CREDIT NOTE (customer + order + invoice + credit note) =====

    "create_full_credit_note": {
        "description": (
            "Create a full credit note flow: create customer, create order with orderLines, "
            "invoice the order, then create a credit note on the invoice.\n"
            "This is used when the prompt says 'Gutschrift', 'kreditnota', 'credit note', 'reklamert' etc. "
            "and there is NO existing invoice — everything must be created from scratch.\n"
            "IMPORTANT: 'comment' is the REASON for the credit note (e.g. 'Reklamasjon', 'Kreditering', 'Gutschrift'). "
            "Extract the reason/complaint from the prompt. If the prompt says 'reklamert' use 'Reklamasjon'."
        ),
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": [
            "customer_name", "customer_email", "customer_organizationNumber",
            "customer_phoneNumber", "customer_phoneNumberMobile",
            "customer_addressLine1", "customer_postalCode", "customer_city",
            "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate",
            "creditNoteDate", "comment",
        ],
        "optimal_calls": 4,
        "steps": [
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                    "phoneNumber": "{{customer_phoneNumber}}",
                    "phoneNumberMobile": "{{customer_phoneNumberMobile}}",
                    "postalAddress": {
                        "addressLine1": "{{customer_addressLine1}}",
                        "postalCode": "{{customer_postalCode}}",
                        "city": "{{customer_city}}",
                    },
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
                "note": "orderLines go IN the order body as array: [{description, count, unitPriceExcludingVatCurrency}]",
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
            {
                "method": "PUT",
                "path": "/invoice/$step_2.id/:createCreditNote",
                "params": {
                    "date": "{{creditNoteDate}}",
                    "comment": "{{comment}}",
                },
                "note": "Creates credit note on the invoice. $step_2 is the invoice action response.",
            },
        ],
    },

    # ===== REVERSE PAYMENT (customer + order + invoice + payment + reverse voucher) =====

    "reverse_payment": {
        "description": (
            "Create a full reverse payment flow: get payment type, create customer, "
            "create order with orderLines, invoice, register payment, then find and reverse the voucher.\n"
            "Used when the prompt says 'reverser betaling', 'reverse payment', 'zurückgebucht', 'stornieren' etc."
        ),
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice", "Voucher"],
        "extract_fields": [
            "customer_name", "customer_email", "customer_organizationNumber",
            "orderLines", "invoiceDate", "invoiceDueDate", "orderDate", "deliveryDate",
            "paymentDate", "paymentAmount", "reverseDate",
        ],
        "optimal_calls": 7,
        "steps": [
            {
                "method": "GET",
                "path": "/invoice/paymentType",
                "params": {"fields": "id,description"},
                "note": "Step 0: Get payment type for registering payment later.",
            },
            {
                "method": "POST",
                "path": "/customer",
                "body": {
                    "name": "{{customer_name}}",
                    "isCustomer": True,
                    "email": "{{customer_email}}",
                    "organizationNumber": "{{customer_organizationNumber}}",
                },
                "note": "Step 1: Create customer.",
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
                "note": "Step 2: Create order with orderLines.",
            },
            {
                "method": "PUT",
                "path": "/order/$step_2.id/:invoice",
                "params": {
                    "invoiceDate": "{{invoiceDate}}",
                    "invoiceDueDate": "{{invoiceDueDate}}",
                    "sendToCustomer": False,
                },
                "note": "Step 3: Invoice the order.",
            },
            {
                "method": "PUT",
                "path": "/invoice/$step_3.id/:payment",
                "params": {
                    "paymentDate": "{{paymentDate}}",
                    "paymentTypeId": "$step_0.values[0].id",
                    "paidAmount": "{{paymentAmount}}",
                },
                "note": "Step 4: Register payment on the invoice.",
            },
            {
                "method": "GET",
                "path": "/ledger/voucher",
                "params": {
                    "dateFrom": "{{paymentDate}}",
                    "dateTo": "{{paymentDatePlusOne}}",
                    "fields": "id,date,description,number",
                },
                "depends_on": [4],
                "note": "Step 5: Find the payment voucher AFTER payment is registered.",
            },
            {
                "method": "PUT",
                "path": "/ledger/voucher/$step_5.values[-1].id/:reverse",
                "params": {
                    "date": "{{reverseDate}}",
                },
                "note": "Step 6: Reverse the last voucher (the payment voucher). reverseDate defaults to paymentDate if not specified.",
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
    "update_employee": ["oppdater ansatt", "endre ansatt", "update employee", "endre telefon", "endre epost", "mitarbeiter aktualisieren", "actualizar empleado", "atualizar empregado", "mettre a jour employe", "oppdater tilsett", "oppdater tilsatt"],
    "create_customer": ["kunde", "customer", "cliente", "client", "Kunde", "opprett kunde", "ny kunde", "registrer kunde"],
    "update_customer": ["oppdater kunde", "endre kunde", "update customer", "kunde aktualisieren", "actualizar cliente", "atualizar cliente", "mettre a jour client"],
    "create_product": ["produkt", "product", "producto", "produto", "Produkt", "produit", "opprett produkt", "nytt produkt", "vare"],
    "create_invoice": ["faktura", "invoice", "factura", "fatura", "Rechnung", "facture", "opprett faktura", "ny faktura"],
    "create_invoice_with_payment": ["faktura med betaling", "invoice with payment", "faktura og betaling"],
    "register_payment": ["innbetaling", "betaling", "payment", "pago", "pagamento", "Zahlung", "paiement", "registrer betaling", "registrer innbetaling", "zahlung registrieren", "registrar pago", "enregistrer paiement", "registrar pagamento"],
    "register_payment_by_search": ["betal faktura nummer", "registrer betaling pa faktura", "payment on invoice number", "betal faktura nr", "betaling for faktura", "pay invoice number", "payment for invoice", "betaling på faktura"],
    "create_credit_note": ["kreditnota", "credit note", "nota de credito", "Gutschrift", "avoir", "note de credit"],
    "create_travel_expense": ["reiseregning", "travel expense", "gastos de viaje", "despesas de viagem", "Reisekosten", "note de frais", "reiserekning", "registrer reiseregning", "registrer reiserekning", "ny reiserekning", "reisekosten erstellen"],
    "delete_travel_expense": ["slett reiseregning", "delete travel", "slett reiserekning", "reisekosten loschen", "reisekosten löschen", "eliminar gasto de viaje", "supprimer note de frais"],
    "deliver_travel_expense": ["lever reiseregning", "deliver travel expense", "send inn reiseregning", "lever reiserekning"],
    "approve_travel_expense": ["godkjenn reiseregning", "approve travel expense", "godkjenn reiserekning"],
    "create_project": ["prosjekt", "project", "proyecto", "projeto", "Projekt", "projet", "opprett prosjekt", "nytt prosjekt"],
    "create_project_existing_customer": ["prosjekt for eksisterende kunde", "project for existing customer", "prosjekt eksisterende"],
    "create_internal_project": ["internt prosjekt", "internal project", "proyecto interno", "internes Projekt", "innvendig prosjekt"],
    "update_project": ["oppdater prosjekt", "endre prosjekt", "update project"],
    "create_department": ["avdeling", "department", "departamento", "Abteilung", "departement", "opprett avdeling", "ny avdeling"],
    "create_supplier": ["leverandør", "leverandor", "supplier", "proveedor", "fornecedor", "Lieferant", "fournisseur", "opprett leverandør", "registrer leverandør", "ny leverandør"],
    "update_supplier": ["oppdater leverandor", "oppdater leverandør", "endre leverandor", "endre leverandør", "update supplier", "lieferant aktualisieren", "actualizar proveedor", "atualizar fornecedor", "mettre a jour fournisseur"],
    "update_department": ["oppdater avdeling", "endre avdeling", "update department"],
    "update_product": ["oppdater produkt", "endre produkt", "update product"],
    "create_contact": ["kontaktperson", "contact person", "persona de contacto", "Kontaktperson", "kontakt"],
    "create_voucher": ["bilag", "voucher", "Beleg", "piece comptable", "postering", "bokfør", "bokfor"],
    "reverse_voucher": ["reverser", "reverse", "tilbakefor"],
    "send_invoice": ["send faktura", "send invoice"],
    "create_supplier_invoice": ["leverandorfaktura", "leverandørfaktura", "supplier invoice", "inngaende faktura", "incoming invoice", "factura proveedor", "Lieferantenrechnung"],
    "create_purchase_order": ["innkjopsordre", "purchase order", "bestilling", "orden de compra", "Bestellung", "bon de commande"],
    "bank_reconciliation": ["bankavstemming", "bank reconciliation", "kontoutskrift", "bank statement", "conciliacion bancaria", "Bankabstimmung"],
    "create_timesheet_entry": ["timeregistrering", "timeforing", "timesheet", "timer", "hours", "horas", "Stunden", "heures"],
    "create_opening_balance": ["apningsbalanse", "opening balance", "inngaende balanse", "balance inicial", "Eroeffnungsbilanz", "eröffnungsbilanz", "balance de apertura", "bilan d'ouverture", "åpningsbalansen"],
    "create_asset": ["anleggsmiddel", "eiendel", "fixed asset", "activo fijo", "Anlagevermoegen"],
    "create_salary_payment": ["lonn", "salary", "loenning", "salario", "Gehalt", "salaire"],
    "create_customer_supplier": ["kunde og leverandor", "kunde og leverandør", "customer and supplier", "both customer and supplier"],
    "create_reminder": ["purring", "reminder", "betalingspaaminnelse", "Zahlungserinnerung", "rappel"],
    "create_employment": ["ansettelse", "employment", "arbeidsforhold", "empleo", "Beschaeftigung"],
    "enable_modules": ["aktiver modul", "enable module", "aktivere", "modul", "module"],
    "create_invoice_and_send": ["opprett og send faktura", "opprett og send ein faktura", "create and send invoice", "crea y envia", "créez et envoyez", "erstellen und senden", "faktura og send"],
    "fixed_price_project_invoice": ["fastpris", "fixed price", "prix forfaitaire", "festpreis", "precio fijo", "fastprisprosjekt", "fixed price project"],
    "create_dimensions_voucher": ["dimensjon", "dimensión contable", "regnskapsdimensjon", "accounting dimension", "custom dimension", "tilpasset dimensjon"],
    "create_full_credit_note": ["gutschrift", "kreditnota", "credit note", "nota de crédito", "nota de credito", "note de crédit", "note de credit", "vollständige gutschrift", "full credit note", "reklamert", "reklamation"],
    "reverse_payment": ["reverser betaling", "reverse payment", "zurückgebucht", "stornieren", "devuelto", "retourné", "annulez paiement", "returnert av banken", "betaling returnert", "payment returned", "payment reversed"],
}

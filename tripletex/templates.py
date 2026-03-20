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
        "extract_fields": ["firstName", "lastName", "email", "dateOfBirth", "phoneNumberMobile", "role", "employeeNumber", "addressLine1", "postalCode", "city"],
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
                "body": {
                    "id": "$step_0.values[0].id",
                    "version": "$step_0.values[0].version",
                    "email": "{{new_email}}",
                },
                "note": "MUST include id and version from GET. Merge with fields_to_update.",
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
        "extract_fields": ["name", "number", "priceExcludingVatCurrency", "priceIncludingVatCurrency", "description"],
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
                },
            },
        ],
    },

    # ===== INVOICING =====

    "create_invoice": {
        "description": "Create an invoice: customer -> order with orderLines -> invoice. NOTE: Company must have bankAccountNumber registered. If 422 about 'bankkontonummer', the sandbox is not properly set up.",
        "relevant_schemas": ["Customer", "Order", "OrderLine", "Invoice"],
        "extract_fields": ["customer_name", "orderLines", "invoiceDate", "invoiceDueDate", "customer_email", "customer_organizationNumber", "customer_phoneNumber", "customer_phoneNumberMobile", "customer_description", "customer_website", "customer_isPrivateIndividual", "customer_addressLine1", "customer_postalCode", "customer_city", "orderDate", "deliveryDate"],
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
                    "description": "{{customer_description}}",
                    "website": "{{customer_website}}",
                    "isPrivateIndividual": "{{customer_isPrivateIndividual}}",
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
        "extract_fields": ["departureDate", "returnDate", "departureFrom", "destination", "purpose", "costs", "isDayTrip", "isForeignTravel", "title", "cost_amount", "cost_description_if_any"],
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
        "description": "Create a supplier invoice (incoming invoice from a supplier). Requires a supplier, an invoice date, and voucher postings. NOTE: dueDate/invoiceDueDate causes errors — do NOT include it.",
        "relevant_schemas": ["Supplier", "Voucher", "Posting"],
        "extract_fields": ["supplier_name", "supplier_organizationNumber", "supplier_email", "supplier_phoneNumber", "supplier_phoneNumberMobile", "supplier_description", "supplier_addressLine1", "supplier_postalCode", "supplier_city", "invoiceNumber", "invoiceDate", "amount", "account_number", "description", "expense_account_number"],
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
                    "supplier": {"id": "$step_0.id"},
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
        "description": "Create a purchase order to a supplier. NOTE: ourContact is REQUIRED — must reference an employee.",
        "relevant_schemas": ["Supplier"],
        "extract_fields": ["supplier_name", "supplier_organizationNumber", "supplier_email", "supplier_phoneNumber", "supplier_phoneNumberMobile", "supplier_description", "supplier_addressLine1", "supplier_postalCode", "supplier_city", "deliveryDate", "orderLines", "ourContact"],
        "optimal_calls": 3,
        "steps": [
            {
                "method": "GET",
                "path": "/employee",
                "params": {"fields": "id,firstName,lastName", "count": 1},
                "note": "Get employee for ourContact (REQUIRED on purchase order).",
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
            },
            {
                "method": "POST",
                "path": "/purchaseOrder",
                "body": {
                    "supplier": {"id": "$step_1.id"},
                    "ourContact": {"id": "$step_0.values[0].id"},
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
            "2. GET /ledger/accountingPeriod to find the accounting period ID for the date range\n"
            "3. POST /bank/reconciliation with accountingPeriod (NOT dateTo — dateTo does NOT exist!)\n"
            "4. For each transaction: POST /bank/reconciliation/match or create vouchers for unmatched items\n"
            "5. If a bank statement (CSV/file) is attached, use POST /bank/statement/import to import it first\n"
            "IMPORTANT: The API does NOT accept dateTo. You MUST use accountingPeriod: {id: X} instead.\n"
            "Use type=MANUAL for manual reconciliation.\n"
            "If the task specifies a closing balance, the sum of matched transactions must equal it."
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
        "description": "Register a fixed asset (anleggsmiddel). NOTE: The date field is 'dateOfAcquisition' (NOT 'acquisitionDate'). Module moduleFixedAssetRegister must be enabled.",
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
        "description": "Create or update employment details for an employee (ansettelsesforhold). NOTE: employmentType and percentageOfFullTimeEquivalent do NOT exist on this endpoint. Only employee and startDate are accepted.",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["search_firstName", "search_lastName", "startDate"],
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
            "customer_name", "customer_email", "customer_organizationNumber", "customer_phoneNumber",
            "customer_phoneNumberMobile", "customer_description", "customer_website", "customer_isPrivateIndividual",
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
                    "description": "{{customer_description}}",
                    "website": "{{customer_website}}",
                    "isPrivateIndividual": "{{customer_isPrivateIndividual}}",
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

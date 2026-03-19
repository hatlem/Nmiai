"""Pre-built plan templates for Tripletex task types.

Each template defines:
- description: What this task type does
- relevant_schemas: Which entity schemas to include in Stage 2 context
- steps: Ordered API calls with {{placeholder}} values for LLM to fill
- extract_fields: Fields the LLM must extract from the prompt
"""

TEMPLATES: dict[str, dict] = {

    # ===== EMPLOYEES =====

    "create_employee": {
        "description": "Create an employee, optionally assign a role/entitlement",
        "relevant_schemas": ["Employee"],
        "extract_fields": ["firstName", "lastName", "email", "dateOfBirth", "phoneNumberMobile", "role"],
        "steps": [
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
                "params": {"employeeId": "$step_0.id", "template": "{{role}}"},
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
        "description": "Create an invoice: customer -> order with orderLines -> invoice",
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
        "description": "Create a project, optionally linked to a customer",
        "relevant_schemas": ["Project", "Customer"],
        "extract_fields": ["name", "customer_name", "startDate", "endDate", "isInternal", "projectManager", "description"],
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
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "customer": {"id": "$step_0.id"},
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
                    "isInternal": False,
                },
            },
        ],
    },

    "create_internal_project": {
        "description": "Create an internal project (no customer)",
        "relevant_schemas": ["Project"],
        "extract_fields": ["name", "startDate", "endDate", "description"],
        "steps": [
            {
                "method": "POST",
                "path": "/project",
                "body": {
                    "name": "{{project_name}}",
                    "isInternal": True,
                    "startDate": "{{startDate}}",
                    "endDate": "{{endDate}}",
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

    # ===== FALLBACK =====

    "unknown": {
        "description": "Task type not recognized — LLM generates plan from scratch using full API reference",
        "relevant_schemas": ["Employee", "Customer", "Product", "Order", "OrderLine", "Invoice", "TravelExpense", "Project", "Department", "Contact", "Supplier", "Voucher", "Posting"],
        "extract_fields": [],
        "steps": [],
    },
}


# Map keywords to task types for fast classification (multilingual)
KEYWORD_HINTS: dict[str, list[str]] = {
    "create_employee": ["ansatt", "employee", "empleado", "empregado", "mitarbeiter", "employe"],
    "update_employee": ["oppdater ansatt", "endre ansatt", "update employee"],
    "create_customer": ["kunde", "customer", "cliente", "client", "Kunde"],
    "create_product": ["produkt", "product", "producto", "produto", "Produkt", "produit"],
    "create_invoice": ["faktura", "invoice", "factura", "fatura", "Rechnung", "facture"],
    "register_payment": ["innbetaling", "betaling", "payment", "pago", "pagamento", "Zahlung", "paiement"],
    "create_credit_note": ["kreditnota", "credit note", "nota de credito", "Gutschrift", "avoir"],
    "create_travel_expense": ["reiseregning", "travel expense", "gastos de viaje", "despesas de viagem", "Reisekosten", "note de frais"],
    "delete_travel_expense": ["slett reiseregning", "delete travel"],
    "create_project": ["prosjekt", "project", "proyecto", "projeto", "Projekt", "projet"],
    "create_department": ["avdeling", "department", "departamento", "Abteilung", "departement"],
    "create_supplier": ["leverandor", "supplier", "proveedor", "fornecedor", "Lieferant", "fournisseur"],
    "create_voucher": ["bilag", "voucher", "Beleg", "piece comptable"],
    "reverse_voucher": ["reverser", "reverse", "tilbakefor"],
    "send_invoice": ["send faktura", "send invoice"],
}

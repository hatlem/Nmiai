# Tripletex API v2 - Komplett Oversikt

> Kilde: [Tripletex API 2.0 Docs](https://tripletex.no/v2-docs/) | [Developer Portal](https://developer.tripletex.no/) | [GitHub](https://github.com/Tripletex/tripletex-api2)
>
> OpenAPI-spesifikasjon: `https://tripletex.no/v2/openapi.json`

---

## Innholdsfortegnelse

1. [Om APIet](#om-apiet)
2. [Autentisering](#autentisering)
3. [API-kategorier og endepunkter](#api-kategorier-og-endepunkter)
4. [Webhooks](#webhooks)
5. [Nyttige lenker](#nyttige-lenker)

---

## Om APIet

Tripletex API v2 er et **RESTful API** for Norges mest brukte regnskapssystem. APIet gir tilgang til å lese, opprette og oppdatere de fleste objekter i ERP-systemet.

**Viktige designprinsipper:**
- Bruker **PUT med valgfrie felter** i stedet for PATCH
- Handlinger/kommandoer prefikses med `:` (f.eks. `/v2/hours/123/:approve`)
- Aggregerte resultater prefikses med `>` (f.eks. `/v2/hours/>thisWeeksBillables`)
- Endepunkter merket **[BETA]** kan endres
- Endepunkter merket **[DEPRECATED]** vil fjernes i fremtidige versjoner

---

## Autentisering

1. Du trenger en **consumer token** og en **employee token**
2. Opprett sesjonstoken via `PUT /token/session/:create` (krever ingen autentisering)
3. Sesjonstokenet brukes i alle påfølgende API-kall

---

## API-kategorier og endepunkter

### 1. Regnskap og Hovedbok (Ledger)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/ledger/account` | GET, POST, PUT, DELETE | Kontoplan - opprett, les og oppdater kontoer |
| `/ledger/posting` | GET | Bokføringsposter |
| `/ledger/postingByDate` | GET | Bokføringsposter etter dato (bedre ytelse) |
| `/ledger/postingRules` | GET, POST, PUT, DELETE | Bokføringsregler |
| `/ledger/vatType` | GET, POST, PUT | MVA-typer |
| `/ledger/vatType/createRelativeVatType` | POST | Opprett relativ MVA-type |
| `/ledger/vatSettings` | GET, PUT | MVA-innstillinger |
| `/ledger/voucher` | GET, POST, PUT, DELETE | Bilag |
| `/ledger/voucher/{id}/:reverse` | PUT | Reverser bilag |
| `/ledger/voucher/{id}/:sendToInbox` | PUT | Send bilag til innboks |
| `/ledger/voucher/>nonPosted` | GET | Ikke-bokførte bilag |
| `/ledger/voucher/>externalVoucherNumber` | GET | Eksternt bilagsnummer |
| `/ledger/voucher/>voucherReception` | GET | Bilagsmottak |
| `/ledger/voucher/list` | POST | Opprett flere bilag |
| `/ledger/voucher/openingBalance` | GET, POST, DELETE | Åpningsbalanse |
| `/ledger/voucher/importDocument` | POST | Importer bilagsdokument |
| `/ledger/voucher/historical/historical` | POST | Historiske bilag |
| `/ledger/voucher/historical/employee` | POST | Historiske bilag for ansatt |
| `/ledger/voucher/{id}/attachment` | POST, DELETE | Vedlegg til bilag |
| `/ledger/paymentTypeOut` | GET, POST, PUT, DELETE | Utgående betalingstyper |

### 2. Faktura (Invoice)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/invoice` | GET, POST | Fakturaer |
| `/invoice/{id}` | GET | Hent enkeltfaktura |
| `/invoice/{id}/:send` | PUT | Send faktura |
| `/invoice/{id}/:createCreditNote` | PUT | Opprett kreditnota |
| `/invoice/{id}/:payment` | POST | Registrer betaling |
| `/invoice/{id}/:createReminder` | PUT | Opprett purring |
| `/invoice/details/{id}` | GET | Fakturadetaljer |
| `/invoice/list` | POST | Opprett flere fakturaer |
| `/invoice/paymentType` | GET | Betalingstyper for faktura |

### 3. Ordre (Order)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/order` | GET, POST, PUT | Ordrer |
| `/order/{id}` | GET, PUT | Enkeltordre |
| `/order/{id}/:invoice` | PUT | Fakturer ordre |
| `/order/{id}/:invoiceMultipleOrders` | PUT | Fakturer flere ordrer |
| `/order/{id}/:attach` | PUT | Legg til vedlegg |
| `/order/{id}/:pickLine` | PUT | Plukk ordrelinje |
| `/order/{id}/:unpickLine` | PUT | Avplukk ordrelinje |
| `/order/{id}/:createCreditNote` | PUT | Opprett kreditnota |
| `/order/{id}/approveSubscriptionInvoice` | PUT | Godkjenn abonnementsfaktura |
| `/order/{id}/unApproveSubscriptionInvoice` | PUT | Avgodkjenn abonnementsfaktura |
| `/order/orderline` | GET, POST, PUT, DELETE | Ordrelinjer |
| `/order/orderGroup` | GET, POST, PUT, DELETE | Ordregrupper |
| `/order/list` | POST | Opprett flere ordrer |

### 4. Innkjøpsordre (Purchase Order)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/purchaseOrder` | GET, POST, PUT, DELETE | Innkjøpsordrer |
| `/purchaseOrder/{id}/:send` | PUT | Send innkjøpsordre |
| `/purchaseOrder/{id}/:sendByEmail` | PUT | Send per e-post |
| `/purchaseOrder/{id}/attachment` | POST, DELETE | Vedlegg |
| `/purchaseOrder/orderline` | GET, POST, PUT, DELETE | Ordrelinjer |
| `/purchaseOrder/goodsReceipt` | GET, POST, PUT, DELETE | Varemottak |
| `/purchaseOrder/goodsReceiptLine` | GET, POST, PUT, DELETE | Varemottakslinjer |
| `/purchaseOrder/deviation` | GET, POST, PUT, DELETE | Avvik |
| `/purchaseOrder/deviation/{id}/:approve` | PUT | Godkjenn avvik |
| `/purchaseOrder/deviation/{id}/:deliver` | PUT | Lever avvik |
| `/purchaseOrder/deviation/{id}/:undeliver` | PUT | Angre levering |

### 5. Leverandørfaktura (Supplier Invoice)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/supplierInvoice` | GET, POST, PUT | Leverandørfakturaer |
| `/supplierInvoice/{id}/:approve` | PUT | Godkjenn |
| `/supplierInvoice/{id}/:reject` | PUT | Avvis |
| `/supplierInvoice/{id}/:addRecipient` | PUT | Legg til mottaker |
| `/supplierInvoice/{id}/:addPayment` | POST | Legg til betaling |
| `/supplierInvoice/{id}/:changeDimension` | PUT | Endre dimensjon |
| `/supplierInvoice/voucher/{id}/postings` | GET | Bilagsposter |

### 6. Kunder (Customer)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/customer` | GET, POST, PUT | Kunder |
| `/customer/{id}` | GET, PUT | Enkeltkunde |
| `/customer/list` | POST | Opprett flere kunder |

### 7. Leverandører (Supplier)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/supplier` | GET, POST, PUT | Leverandører |
| `/supplier/{id}` | GET, PUT | Enkeltleverandør |
| `/supplier/list` | POST | Opprett flere leverandører |

### 8. Kontakter (Contact)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/contact` | GET, POST, PUT | Kontaktpersoner |
| `/contact/list` | POST, DELETE | Opprett/slett flere kontakter |

### 9. Adresser (Address)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/address` | GET, PUT | Adresser |
| `/deliveryAddress` | GET, POST, PUT | Leveringsadresser |

### 10. Produkter (Product)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/product` | GET, POST, PUT | Produkter |
| `/product/{id}` | GET, PUT, DELETE | Enkeltprodukt |
| `/product/{id}/image` | POST, DELETE | Produktbilde |
| `/product/list` | POST | Opprett flere produkter |
| `/product/external` | GET, POST, PUT, DELETE | Eksterne produkter |
| `/product/external/query` | GET | Søk eksterne produkter |
| `/product/unit` | GET, POST, PUT | Enheter |
| `/product/unit/master` | GET | Masterenheter |
| `/product/price` | GET | Produktpriser |
| `/product/productPrice` | GET, POST, PUT | Produktpris-innstillinger |
| `/product/discountGroup` | GET, POST, PUT, DELETE | Rabattgrupper |
| `/product/group` | GET, POST, PUT, DELETE | Produktgrupper |
| `/product/groupRelation` | GET, POST, DELETE | Produktgruppe-relasjoner |
| `/product/inventoryLocation` | GET, POST, PUT, DELETE | Lagerlokasjon |
| `/product/logisticsSettings` | GET, PUT | Logistikkinnstillinger |

### 11. Lager og Inventar (Inventory)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/inventory` | GET | Lagerbeholdning |
| `/inventory/inventories` | GET | Alle lagre |
| `/inventory/location` | GET, POST, PUT, DELETE | Lagerlokasjoner |
| `/inventory/inventoryLocation` | GET | Lagerlokasjon-kobling |
| `/inventory/stocktaking` | GET, POST, PUT, DELETE | Varetelling |
| `/inventory/stocktaking/productline/{id}/:changeLocation` | PUT | Endre lokasjon for varetelling |

### 12. Prosjekter (Project)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/project` | GET, POST, PUT, DELETE | Prosjekter |
| `/project/{id}` | GET, PUT | Enkeltprosjekt |
| `/project/{id}/period/hourlistReport` | GET | Timeliste-rapport |
| `/project/{id}/period/invoiced` | GET | Fakturert per periode |
| `/project/{id}/period/invoicingReserve` | GET | Faktureringsreserve |
| `/project/{id}/period/monthlyStatus` | GET | Månedlig status |
| `/project/{id}/period/overallStatus` | GET | Totalstatus |
| `/project/task` | GET, POST, PUT | Prosjektoppgaver |
| `/project/participant` | GET, POST, PUT, DELETE | Prosjektdeltakere |
| `/project/orderline` | GET, POST, PUT, DELETE | Prosjektordrelinjer |
| `/project/hourlyRates` | GET, POST, PUT, DELETE | Timepriser |
| `/project/hourlyRates/projectSpecificRates` | GET, POST, PUT, DELETE | Prosjektspesifikke timepriser |
| `/project/resourcePlanBudget` | GET, POST, PUT | Ressursplanbudsjett |
| `/project/projectActivity` | GET, POST, DELETE | Prosjektaktiviteter |
| `/project/controlForm` | GET, POST, PUT | Kontrollskjema |
| `/project/controlFormType` | GET, POST, PUT, DELETE | Kontrollskjematyper |
| `/project/settings` | GET, PUT | Prosjektinnstillinger |
| `/project/template/{id}` | GET | Prosjektmaler |
| `/project/category` | GET, POST, PUT | Prosjektkategorier |

### 13. Ansatte (Employee)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/employee` | GET, POST, PUT | Ansatte |
| `/employee/{id}` | GET, PUT | Enkeltansatt |
| `/employee/employment` | GET, POST, PUT | Ansettelsesforhold |
| `/employee/employment/details` | GET, POST, PUT | Ansettelsesdetaljer |
| `/employee/employment/occupationCode` | GET | Yrkeskoder |
| `/employee/employment/employmentType` | GET | Ansettelsestyper |
| `/employee/employment/leaveOfAbsence` | GET, POST, PUT | Permisjoner |
| `/employee/employment/leaveOfAbsenceType` | GET | Permisjonstyper |
| `/employee/employment/remunerationType` | GET | Lønnstyper |
| `/employee/employment/workingHoursScheme` | GET | Arbeidstidsordninger |
| `/employee/standardTime` | GET, POST, PUT | Standard arbeidstid |
| `/employee/standardTime/byDate` | GET | Arbeidstid per dato |
| `/employee/hourlyCostAndRate` | GET, POST, PUT | Timekost og -sats |
| `/employee/nextOfKin` | GET, POST, PUT | Pårørende |
| `/employee/preferences` | GET, PUT | Ansattpreferanser |
| `/employee/category` | GET, POST, PUT, DELETE | Ansattkategorier |
| `/employee/entitlement` | GET | Rettigheter |

### 14. Lønn (Salary)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/salary/payslip` | GET | Lønnslipper |
| `/salary/specification` | GET | Lønnsspesifikasjoner |
| `/salary/transaction` | GET, POST, DELETE | Lønnstransaksjoner |
| `/salary/transaction/{id}/attachment` | POST, DELETE | Vedlegg til transaksjon |
| `/salary/type` | GET | Lønnstyper |
| `/salary/compilation` | GET | Lønnssammenstilling |
| `/salary/compilation/pdf` | GET | Lønnssammenstilling PDF |
| `/salary/settings/standardTime` | GET, POST, PUT | Standardtid-innstillinger |
| `/salary/settings/pensionScheme` | GET, POST, PUT, DELETE | Pensjonsordninger |

### 15. Timer og Timeføring (Timesheet)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/timesheet/entry` | GET, POST, PUT, DELETE | Timeregistreringer |
| `/timesheet/month` | GET | Månedsoversikt |
| `/timesheet/settings` | GET | Timeinnstillinger |
| `/timesheet/salaryTypeSpecification` | GET, POST, PUT, DELETE | Lønnstype-spesifikasjoner |
| `/timesheet/timeClock` | GET, POST, PUT | Stemplingsur |

### 16. Reiseregninger (Travel Expense)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/travelExpense` | GET, POST, PUT, DELETE | Reiseregninger |
| `/travelExpense/{id}/attachment` | POST, DELETE | Vedlegg |
| `/travelExpense/:deliver` | PUT | Lever reiseregning |
| `/travelExpense/:approve` | PUT | Godkjenn reiseregning |
| `/travelExpense/:copy` | PUT | Kopier reiseregning |
| `/travelExpense/cost` | GET, POST, PUT, DELETE | Kostnader |
| `/travelExpense/costCategory` | GET | Kostnadskategorier |
| `/travelExpense/paymentType` | GET | Betalingstyper |
| `/travelExpense/perDiemCompensation` | GET, POST, PUT, DELETE | Diettgodtgjørelse |
| `/travelExpense/perDiemCompensation/autoSuggest` | GET | Automatisk forslag diett |
| `/travelExpense/zone` | GET | Soner |
| `/travelExpense/settings` | GET | Reiseregningsinnstillinger |
| `/travelExpense/rate` | GET | Satser |
| `/travelExpense/rateCategory` | GET | Satskategorier |
| `/travelExpense/rateCategoryGroup` | GET | Satskategorigrupper |

### 17. Aktiviteter (Activity)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/activity` | GET, POST | Aktiviteter |
| `/activity/{id}` | GET | Enkeltaktivitet |
| `/activity/list` | POST | Opprett flere aktiviteter |

### 18. Eiendeler (Asset)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/asset` | GET, POST, PUT, DELETE | Anleggsmidler |
| `/asset/{id}/postings` | GET | Bokføringsposter for eiendel |
| `/asset/upload` | POST | Importer eiendeler (Excel) |

### 19. Bank og Betalinger

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/bank` | GET | Bankkontoer |
| `/bank/statement/import` | POST | Importer kontoutskrift |
| `/bank/reconciliation` | GET, POST, PUT, DELETE | Bankavstemming |
| `/bank/reconciliation/match` | GET, POST, PUT, DELETE | Avstemmingsmatch |

### 20. Selskap (Company)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/company` | GET, PUT | Selskapsinformasjon |
| `/company/settings/altinn` | GET, PUT | Altinn-innstillinger |
| `/company/divisions` | GET | Avdelinger |
| `/company/salesmodules` | GET | Salgsmoduler |

### 21. Avdeling (Department)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/department` | GET, POST, PUT, DELETE | Avdelinger |
| `/department/{id}` | GET | Enkeltavdeling |
| `/department/list` | POST | Opprett flere avdelinger |

### 22. Divisjon (Division)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/division` | GET, POST, PUT | Divisjoner |
| `/division/list` | POST | Opprett flere divisjoner |

### 23. Valuta (Currency)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/currency` | GET | Valutaer |
| `/currency/{id}` | GET | Enkeltvaluta |
| `/currency/{id}/rate` | GET | Valutakurser |

### 24. Finansiell Rapportering

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/balanceSheet` | GET | Balanserapport |
| `/resultbudget` | GET | Resultatbudsjett |
| `/resultbudget/company` | GET | Resultatbudsjett selskap |
| `/resultbudget/department/{id}` | GET | Resultatbudsjett avdeling |
| `/resultbudget/project/{id}` | GET | Resultatbudsjett prosjekt |
| `/resultbudget/product/{id}` | GET | Resultatbudsjett produkt |
| `/resultbudget/employee/{id}` | GET | Resultatbudsjett ansatt |

### 25. Dokumenter og Arkiv

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/document` | GET, POST | Dokumenter |
| `/documentArchive/project/{id}` | GET, POST | Prosjektarkiv |
| `/documentArchive/employee/{id}` | GET, POST | Ansattarkiv |
| `/documentArchive/customer/{id}` | GET, POST | Kundearkiv |
| `/documentArchive/supplier/{id}` | GET, POST | Leverandørarkiv |
| `/documentArchive/product/{id}` | GET, POST | Produktarkiv |
| `/documentArchive/account/{id}` | GET, POST | Kontoarkiv |
| `/documentArchive/prospect/{id}` | GET, POST | Prospektarkiv |
| `/documentArchive/{id}` | GET, PUT, DELETE | Enkeltdokument |

### 26. Bilagsinnboks og Status

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/voucherInbox/inboxCount` | GET | Antall bilag i innboks |
| `/voucherStatus` | GET | Bilagsstatus |
| `/voucherMessage` | GET, POST | Bilagsmeldinger |

### 27. Purring (Reminder)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/reminder` | GET | Purringer |
| `/reminder/{id}/pdf` | GET | Last ned purring som PDF |

### 28. SAF-T Import/Eksport

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/saft/exportSAFT` | GET | Eksporter SAF-T fil |
| `/saft/importSAFT` | POST | Importer SAF-T fil |

### 29. CRM og Prospekter

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/crm/prospect` | GET, POST, PUT | Prospekter/leads |
| `/crm/prospect/{id}` | GET, PUT | Enkeltprospekt |

### 30. Webhooks og Events

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/event` | GET | Hendelsestyper |
| `/event/{eventType}` | GET | Spesifikk hendelsestype |
| `/event/subscription` | GET, POST, PUT, DELETE | Webhook-abonnementer |
| `/event/subscription/list` | POST | Opprett flere abonnementer |

### 31. Token og Sesjon

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/token/session/:create` | PUT | Opprett sesjonstoken |
| `/token/session/{token}` | DELETE | Slett sesjonstoken |
| `/token/consumer/byToken` | GET | Hent forbrukertoken |

### 32. Logistikk og Levering

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/pickupPoint` | GET | Hentepunkter |
| `/pickupPoint/{id}` | GET | Enkelt hentepunkt |
| `/transportType` | GET | Transporttyper |
| `/discountPolicy` | GET | Rabattreguler |

### 33. Land og Kommune

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/country` | GET | Land |
| `/municipality/query` | GET | Kommuner |

### 34. Pensjon

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/pension` | GET | Pensjonsinformasjon |

### 35. Årsoppgjør (Year End)

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/yearEnd` | GET | Årsoppgjør |
| `/yearEnd/enumType/businessActivityTypes` | GET | Virksomhetstyper |

### 36. MVA-innstillinger

| Endepunkt | Metoder | Beskrivelse |
|---|---|---|
| `/vatTermSizeSettings` | GET, PUT | MVA-termin innstillinger |

---

## Webhooks

Tripletex støtter webhooks for å motta sanntidsvarsler om hendelser. Tilgjengelige hendelsestyper inkluderer blant annet:

- `order.create` / `order.update` / `order.delete`
- `product.create` / `product.update` / `product.delete`
- `customer.create` / `customer.update` / `customer.delete`
- `supplier.create` / `supplier.update` / `supplier.delete`
- `invoice.create` / `invoice.update` / `invoice.delete`
- `project.create` / `project.update` / `project.delete`
- `employee.create` / `employee.update`
- `voucher.create` / `voucher.update` / `voucher.delete`
- `purchaseOrder.create` / `purchaseOrder.update` / `purchaseOrder.delete`
- `supplierInvoice.voucher.posted`

Se `/event` og `/event/subscription` endepunktene for fullstendig liste.

---

## Nyttige lenker

| Ressurs | URL |
|---|---|
| API-dokumentasjon (produksjon) | https://tripletex.no/v2-docs/ |
| API-dokumentasjon (test) | https://api-test.tripletex.tech/v2-docs/ |
| OpenAPI-spesifikasjon | https://tripletex.no/v2/openapi.json |
| Developer Portal | https://developer.tripletex.no/ |
| GitHub (eksempler + changelog) | https://github.com/Tripletex/tripletex-api2 |
| Changelog | https://github.com/Tripletex/tripletex-api2/blob/master/changelog.md |
| FAQ | https://github.com/Tripletex/tripletex-api2/blob/master/FAQ.md |

---

**Merk:** Denne oversikten er basert på offentlig tilgjengelig dokumentasjon og changelog per mars 2026. For den mest oppdaterte og komplette listen, se den interaktive OpenAPI-dokumentasjonen på https://tripletex.no/v2-docs/ eller last ned spesifikasjonen fra https://tripletex.no/v2/openapi.json.

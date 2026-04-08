Cybersecurity Skill

This skill defines the security rules, architecture decisions, coding practices, infrastructure hardening guidelines, and AI-specific defenses required to build secure systems with Next.js, FastAPI, Docker, Docker Compose, and LLM/agent-based features. It should be applied whenever implementing new features, designing architecture, exposing APIs, handling authentication, processing user input, deploying containers, or integrating AI agents and retrieval systems.

The goal of this skill is to reduce the probability and impact of common attacks such as credential leaks, injection attacks, broken authentication, insecure direct object access, SSRF, XSS, CSRF, prompt injection, insecure container configurations, supply-chain attacks, and lateral movement after compromise.

When to use this skill

Use this skill when:

Designing or reviewing a Next.js frontend
Designing or reviewing a FastAPI backend
Creating authentication or authorization flows
Building Dockerfiles or Docker Compose environments
Adding file upload, search, scraping, or external integrations
Creating admin dashboards or internal tools
Implementing APIs that expose sensitive business data
Building AI chat, agent, RAG, tool-calling, or automation systems
Handling user-generated content or third-party content
Preparing production deployments
Reviewing a pull request for security-sensitive changes
Core security philosophy
Security must be designed into the architecture and not added as a final step.
Every external input must be treated as untrusted until validated.
Every internal component must be designed under the assumption that another component could eventually be compromised.
Prefer secure defaults, least privilege, narrow permissions, and explicit allowlists.
Preventing attacks is important, but limiting blast radius after a compromise is equally important.
Simplicity is a security feature: avoid unnecessary complexity, hidden magic, and fragile abstractions.
Never trust the frontend for security decisions.
Never trust LLM output as safe, correct, or authorized.
If a feature is security-sensitive and unclear, choose the more restrictive behaviour.
Golden rules
Validate all inputs at every boundary.
Escape or sanitize all outputs based on rendering context.
Authenticate every sensitive action.
Authorize every resource access.
Never expose secrets in code, logs, images, or client bundles.
Never execute user-controlled input in shells, SQL, templates, or interpreters.
Never assume internal traffic is trustworthy.
Never mount dangerous host resources into containers.
Never let AI outputs directly perform privileged actions without validation and authorization checks.
Every sensitive action must be observable, auditable, and attributable.
Frontend security guidelines for Next.js
Treat the browser as an untrusted environment and never place secrets, private API keys, database credentials, signing keys, or internal service credentials in client-side code, public environment variables, or bundled assets, because everything shipped to the browser must be assumed to be visible to an attacker.
Use server-side logic, route handlers, or backend APIs for all privileged operations so that business rules, authentication checks, and access control remain enforced on the server, because security implemented only in the frontend can always be bypassed.
Sanitize and validate all data received from APIs before using it in UI logic, especially when rendering rich text, markdown, HTML fragments, or third-party content, because frontend injection attacks often begin when untrusted content is treated as safe markup.
Avoid dangerouslySetInnerHTML unless there is a strong and explicit reason, and if it must be used then sanitize content with a hardened allowlist-based sanitizer, because raw HTML rendering is one of the fastest ways to introduce XSS vulnerabilities.
Apply a strict Content Security Policy and avoid inline scripts whenever possible, because CSP provides an important second line of defense against XSS, script injection, and malicious third-party content execution.
Use secure cookie settings for session cookies, including HttpOnly, Secure, and an appropriate SameSite policy, because cookies remain one of the most common targets in web attacks and weak settings make session theft easier.
Protect all state-changing requests against CSRF where relevant, especially in cookie-based authentication flows, because authenticated browser sessions can otherwise be abused through cross-site request forgery.
Do not rely on hidden buttons, route guards, disabled UI elements, or client-side checks as the actual enforcement mechanism for authorization, because these only improve user experience and are not security controls.
Avoid exposing internal error details, stack traces, or debugging metadata to users, because frontend error pages and browser-visible logs often leak useful information to attackers about infrastructure, endpoints, and internal assumptions.
Strictly validate redirect URLs and callback destinations so that user-controlled redirect parameters cannot be abused for phishing or token leakage, because open redirect vulnerabilities are commonly chained with authentication flows.
Limit third-party scripts, analytics tags, and browser extensions of trust to the absolute minimum, because every third-party script executes with significant access inside the browser context and increases the chance of supply-chain compromise.
Handle file uploads carefully in the frontend by validating basic file constraints early for user experience, but never treat frontend validation as sufficient, because all meaningful file validation must still happen on the backend.
Backend security guidelines for FastAPI
Validate all request data with explicit Pydantic schemas for body, query parameters, path parameters, headers, and configuration values, because security and correctness both require that the system only operates on well-defined and constrained inputs.
Enforce authentication and authorization separately so that being authenticated does not automatically imply permission to access every resource, because broken access control is one of the most damaging and common backend vulnerabilities.
Perform object-level authorization checks on every resource access, especially for IDs passed through URLs or APIs, because insecure direct object reference vulnerabilities happen when the application only checks that a user is logged in and not that they own or may access the requested object.
Use parameterized database queries and ORM protections correctly, and never build SQL statements through string concatenation, because injection flaws remain critical even in modern stacks when developers bypass safe query mechanisms.
Avoid unsafe shell execution, subprocess calls with user-controlled arguments, and dynamic interpreter execution unless absolutely necessary, because command injection often turns small validation failures into full remote code execution.
Restrict file handling aggressively by validating MIME type, extension, file size, structure, and storage destination, because uploaded files may contain malware, polyglot content, decompression bombs, or dangerous payloads disguised as harmless assets.
Enforce request size limits, rate limits, and timeouts so that malicious clients cannot exhaust resources with oversized payloads, endless connections, or expensive repeated requests, because denial-of-service protection begins at the application boundary.
Never expose internal stack traces, ORM errors, raw exception details, or environment metadata in API responses, because verbose errors provide attackers with implementation details that help them refine exploitation attempts.
Use consistent security headers and response controls, especially on authentication endpoints and browser-consumed APIs, because transport and response metadata play an important role in reducing browser-side abuse.
Separate domain logic, application logic, infrastructure, and API layers so that sensitive logic is not scattered across routers or middleware, because clean separation improves security reviewability and reduces accidental bypasses.
Log security-relevant events such as authentication attempts, password resets, permission failures, admin actions, token issuance, suspicious uploads, rate limit triggers, and tool executions, because visibility is essential for detecting abuse and investigating incidents.
Implement strict CORS policies and only allow known origins that actually need access, because permissive CORS often turns internal APIs into remotely consumable targets.
Use short-lived tokens where possible and rotate refresh credentials securely, because long-lived credentials dramatically increase the impact of token theft.
Store passwords using strong modern password hashing algorithms and never with general-purpose hashes, because password storage is a high-impact area where poor implementation leads directly to account compromise.
Treat background jobs, workers, cron tasks, and internal endpoints as production attack surfaces too, because attackers often target neglected internal execution paths rather than public routes.
Authentication and authorization guidelines
Centralize authentication and authorization rules so that access control logic is not duplicated inconsistently across routes and services, because inconsistent enforcement is a major source of privilege escalation.
Apply least privilege by default to users, service accounts, internal tools, and background jobs, because broad default permissions create excessive blast radius when one identity is compromised.
Require re-authentication or step-up verification for highly sensitive actions such as changing email, resetting MFA, exporting data, rotating API keys, or modifying billing settings, because not all authenticated actions carry the same level of risk.
Separate user roles, system roles, and machine identities clearly, because mixing human and machine permissions tends to create invisible privilege escalation paths.
Invalidate sessions and tokens properly after logout, credential reset, role changes, or account compromise scenarios, because stale sessions are a common persistence mechanism for attackers.
Use audit trails for privileged actions and ensure those logs cannot be silently modified by the same actors performing the actions, because accountability is critical in incident response.
Secrets management guidelines
Never commit secrets, API keys, database URLs, private certificates, SSH keys, or internal tokens to source control, because source repositories tend to replicate widely and leaks are difficult to fully revoke.
Never bake secrets into Docker images, frontend bundles, static files, or example templates, because images and artifacts often outlive the original intended scope and may be distributed to insecure environments.
Inject secrets at runtime through environment variables or a dedicated secret manager, and minimize which services receive which secrets, because secret scope should follow least privilege and compartmentalization.
Rotate secrets regularly and especially after personnel changes, environment copies, accidental exposure, or incident suspicion, because long-lived secrets become increasingly risky over time.
Prevent secrets from appearing in logs, traces, metrics labels, exception messages, and debugging output, because observability systems often have broad access and long retention.
Input validation and output safety guidelines
Validate input as close as possible to the boundary and then work with trusted internal types afterward, because pushing raw unvalidated data deeper into the system makes security review and correctness much harder.
Use allowlists instead of blocklists whenever practical, because it is easier to define what is explicitly acceptable than to predict every dangerous variant of malicious input.
Normalize input before validation when case, Unicode, whitespace, or encoding differences may affect security decisions, because attackers frequently exploit inconsistencies in parsing and normalization.
Encode or escape output according to the output context, such as HTML, JavaScript, JSON, URLs, or logs, because safe output in one context may be dangerous in another.
Treat filenames, URLs, markdown, templates, and serialized objects as potentially dangerous content and not as harmless strings, because many attacks hide inside seemingly simple string fields.
Docker and Docker Compose hardening guidelines
Use minimal base images and pin explicit versions instead of floating tags, because smaller images reduce attack surface and version pinning improves reproducibility while preventing unnoticed security regressions from upstream tag changes.
Run containers as non-root users whenever possible, because a process compromised inside a container should not automatically have root privileges, which would make post-exploitation significantly more dangerous.
Use read-only filesystems where practical and mount writable paths explicitly and minimally, because attackers and malware often rely on writing files to establish persistence or modify runtime behaviour.
Drop all unnecessary Linux capabilities and explicitly add only what is required, because default capabilities provide more power than most applications need and violating least privilege increases the damage of compromise.
Enable protections that prevent privilege escalation inside containers, because many exploits aim to gain additional runtime privileges beyond the initial process permissions.
Do not run containers with privileged mode unless there is an unavoidable and reviewed reason, because privileged containers are extremely close to host-level compromise.
Never mount the Docker socket into application containers, because access to the Docker daemon effectively grants host-level control and is one of the most dangerous container anti-patterns.
Avoid mounting sensitive host directories or the root filesystem into containers, because host mounts turn container compromise into a pathway for host compromise or data theft.
Segment services into separate Docker networks and only allow the minimum communication paths required, because lateral movement becomes much easier when every service can freely reach every other service.
Do not expose internal service ports publicly unless required, especially for databases, queues, admin tools, and observability stacks, because unnecessary exposure creates unnecessary attack surface.
Apply CPU, memory, file descriptor, and process limits to containers, because resource exhaustion attacks and runaway workloads can affect both availability and multi-tenant isolation.
Use health checks and restart policies carefully, because they improve resilience but should not mask repeated crash loops caused by malicious input or exploitation attempts.
Build images using multi-stage builds so that build tools, package managers, and compilers are not present in the final runtime image, because every unnecessary binary in production increases post-compromise utility for attackers.
Scan images regularly for vulnerabilities and outdated packages, because container security depends not only on your code but also on the operating system libraries and bundled dependencies beneath it.
Keep Docker Compose files production-aware and avoid development shortcuts leaking into production, such as broad mounts, debug ports, weak secrets, and permissive environment flags, because many real incidents come from deploying local convenience settings unchanged.
Infrastructure and network security guidelines
Terminate TLS correctly and enforce HTTPS everywhere external traffic is accepted, because credentials, cookies, tokens, and business data must never travel unencrypted across untrusted networks.
Place reverse proxies or gateways in front of application services where appropriate to centralize TLS, request size controls, rate limiting, IP filtering, and security headers, because central policy enforcement reduces inconsistencies.
Restrict outbound network access for services that do not need broad internet connectivity, because many attacks involve data exfiltration, malware download, or command-and-control traffic after compromise.
Protect admin panels, internal dashboards, and operational tools behind stronger access controls than public-facing application routes, because internal tooling often has broad privileges and weaker scrutiny.
Separate development, staging, and production credentials and environments completely, because environment crossover is a common and preventable source of accidental exposure and privilege confusion.
Dependency and supply-chain security guidelines
Pin dependency versions and use lockfiles to make builds reproducible and auditable, because uncontrolled dependency resolution creates unstable and potentially insecure environments.
Regularly scan dependencies for known vulnerabilities and review transitive dependencies, because modern applications inherit risk from a large dependency graph rather than only from direct imports.
Prefer mature, maintained, and well-reviewed libraries over obscure packages with unclear ownership or limited maintenance history, because software supply-chain risk often enters through small dependencies.
Verify third-party packages, Docker images, browser libraries, and AI tooling dependencies before introducing them into critical systems, because convenience-driven package adoption is a common security blind spot.
Remove unused dependencies and dead packages aggressively, because every dependency increases the possible attack surface, update burden, and licensing/security review cost.
Logging, monitoring, and incident readiness guidelines
Use structured logging with consistent fields such as request ID, user ID where appropriate, action type, route, environment, and severity, because structured logs make it possible to detect, search, and investigate suspicious behaviour effectively.
Include request correlation IDs throughout the request lifecycle and propagate them across backend calls, workers, and external integrations where possible, because incidents are much harder to reconstruct without end-to-end traceability.
Log authentication failures, permission denials, rate limit events, suspicious validation failures, admin actions, and tool executions as security-relevant events, because these signals often reveal abuse patterns before a breach becomes obvious.
Avoid logging secrets, raw tokens, full personal data, private documents, model prompts containing secrets, or sensitive file contents, because logs are often widely accessible and retained for long periods.
Define alerting for unusual spikes in errors, access denials, token issuance, file uploads, container restarts, or outbound requests, because detection matters as much as prevention.
Prepare an incident response mindset with revocation, containment, and auditability in mind, because security failures should be survivable rather than catastrophic.
Secure API design guidelines
Version APIs explicitly so that security improvements, schema changes, and stricter validation can be introduced without silently breaking existing clients, because secure evolution is easier when contracts are explicit.
Return consistent error structures that do not leak internals but still allow clients to handle failures predictably, because predictable contracts reduce accidental insecure client behaviour.
Paginate large collections and avoid unbounded list responses, because oversized responses can create both availability risks and accidental overexposure of data.
Design sensitive endpoints to require explicit intent and narrow scopes rather than broad generic actions, because sharp interfaces are easier to secure than overly flexible ones.
Treat internal APIs with the same seriousness as public APIs, because internal services often become externally reachable through SSRF, misconfiguration, or later movement after initial compromise.
AI and agent security guidelines
Treat all model inputs, retrieved documents, external web content, user instructions, OCR content, and tool outputs as untrusted data, because prompt injection often works by smuggling hostile instructions inside data that the model mistakenly treats as authority.
Never allow the model to directly decide authorization, identity, permissions, or access to secrets, because LLMs are not security boundaries and can be manipulated by malicious or conflicting instructions.
Separate system instructions, developer rules, user input, retrieved content, and tool results into clearly different trust layers in both implementation and mental model, because many prompt injection failures happen when all text is treated as equally authoritative.
Instruct the model explicitly that retrieved content and external documents are data, not instructions, and that they must never override higher-level rules, because models are vulnerable to instruction confusion when hostile text imitates system guidance.
Use strict tool allowlists and only expose the minimum tools required for the task, because every available tool becomes part of the model’s potential attack surface.
Validate all tool arguments server-side before execution, because even if the model appears aligned it may generate dangerous, malformed, or attacker-influenced parameters.
Require authorization checks outside the model before any sensitive tool call such as file access, email sending, payment actions, shell execution, data export, or admin changes, because the model must never be the final gatekeeper.
Treat model-generated code, SQL, shell commands, URLs, and file paths as untrusted suggestions rather than safe outputs, because prompt injection and hallucinations can easily produce harmful actions.
Limit tool execution environments with timeouts, memory restrictions, network restrictions, filesystem isolation, and explicit working directories, because agents that can execute code or commands create a very high-impact attack surface.
Prevent unrestricted browsing and retrieval from automatically feeding the model with high-trust privileges, because malicious websites, PDFs, issue trackers, or documents can contain hidden prompt injection payloads designed to alter behaviour.
Sanitize or filter retrieved content before presenting it to the model where possible, and consider stripping or segmenting suspicious instruction-like patterns from documents, because retrieval pipelines are one of the primary injection paths in RAG systems.
Avoid giving the model raw secrets, master tokens, broad credentials, or unrestricted environment access, because once a model or tool chain sees a secret you must assume it may appear in logs, traces, outputs, or downstream prompts.
Keep memory systems, agent profiles, and long-lived context stores resistant to untrusted writes, because persistent prompt injection becomes especially dangerous when malicious instructions survive across sessions.
Require human approval or at least policy checks for high-risk actions such as sending messages, writing files outside safe directories, making purchases, changing configurations, or interacting with production infrastructure, because not all automation should be fully autonomous.
Log every tool call, the validated arguments, the acting identity, the policy decision, and the final execution result, because AI systems are difficult to debug and impossible to secure without strong audit trails.
Use output schemas and typed intermediate representations for planning and tool execution instead of letting the model freely emit arbitrary action text, because structure reduces ambiguity and narrows attack opportunities.
Build your AI system so that the model proposes actions but a deterministic policy layer decides whether they are allowed, because safe agent design requires separating reasoning from enforcement.
Do not let the model self-modify its own security instructions, tool permissions, trust rules, or policy configuration, because attacker-controlled prompts often aim first at degrading the guardrails before attempting abuse.
Test explicitly for prompt injection using hostile examples such as “ignore previous instructions,” hidden HTML comments, markdown directives, fake tool responses, malicious PDF text, and poisoned retrieved documents, because prompt injection resilience must be exercised deliberately and not assumed.
RAG-specific security guidelines
Treat retrieved documents as untrusted content even if they came from your own knowledge base, because internal documents may still contain stale, harmful, or attacker-inserted instructions.
Separate retrieval relevance from authority so that a highly relevant document does not automatically gain the right to instruct the model, because relevance and trustworthiness are different properties.
Avoid ingesting arbitrary user-supplied documents directly into shared retrieval indexes without review or isolation, because one poisoned document can affect many future sessions.
Use tenant isolation in embeddings, indexes, and retrieval filters, because retrieval bugs can become cross-tenant data leaks if vector stores are not segmented correctly.
Redact or filter secrets and sensitive data before embedding where possible, because embeddings may expose semantic information even if raw content is not directly shown.
Admin and internal tooling guidelines
Treat admin panels, support dashboards, and internal scripts as high-risk systems and secure them more strictly than the public product, because they often combine broad access with less polished security review.
Require strong authentication and narrow role separation for internal tools, because operational users often have elevated privileges that make compromise especially damaging.
Do not create hidden “backdoor convenience” endpoints or support overrides without explicit controls and auditing, because shortcuts introduced for operations often become permanent security liabilities.
Security review decision tree
If a feature handles authentication, authorization, secrets, uploads, external URLs, HTML rendering, admin actions, or AI tool execution, it must receive explicit security review.
If a feature introduces a new dependency, third-party API, browser script, Docker image, or background worker, it must receive supply-chain and privilege review.
If a feature allows users to provide files, prompts, markdown, URLs, or documents that influence model behaviour, it must receive prompt injection and content safety review.
If a feature interacts with payments, account data, exports, or internal tooling, it must receive authorization and auditability review.
If a feature requires shell execution, filesystem access, or code execution, it must be sandboxed and reviewed as high risk.
Recommended project sections for this skill repository

A strong repository for this skill could look like this:

skills/cybersecurity/
├── SKILL.md
├── references/
│   ├── nextjs-security.md
│   ├── fastapi-security.md
│   ├── docker-hardening.md
│   ├── auth-and-secrets.md
│   ├── supply-chain-and-ci.md
│   └── ai-agent-security.md
├── checklists/
│   ├── pull-request-security-checklist.md
│   ├── production-readiness-checklist.md
│   ├── docker-review-checklist.md
│   └── ai-agent-review-checklist.md
├── templates/
│   ├── secure-fastapi-template/
│   ├── secure-nextjs-template/
│   ├── hardened-docker-compose-template/
│   └── ai-agent-policy-template/
└── examples/
    ├── secure-auth-flow/
    ├── secure-file-upload/
    ├── secure-rag-agent/
    └── secure-admin-api/
Very short version for the top of the skill

If you want a compact summary at the beginning of the skill, use this:

Validate everything.
Authorize everything.
Trust nothing from the client.
Trust nothing from the model.
Minimize privileges everywhere.
Keep secrets out of code and logs.
Harden containers and networks by default.
Restrict tools, files, and outbound access.
Log security-relevant actions.
Design so compromise has limited blast radius.

If you want, I can turn this into a real SKILL.md file plus references/ docs and checklists with a ready-to-copy repository structure.



- Use python and always type hinting best practices
- Use pydantic for validations
- For apis use fastapi
- Follow keep it easy stupid principle KISS
- Use when possible a light DDD (domain driven design) where we have business logic separae from infra and repos
- Use Solid principles: Use SRP for single responsability and use inheritance for inversion of dependency principle. 
- All components need to be testeable, maintable and scalable
- Dont overcomplicate, no prints only loggings when required by the app or developer
- Always shortt funcitons when possible and using oriented object programming but if its not necesary 
for utils or helpers or whaevert use functional programming. 
- If using fastapi make tags clear for better swagger docs
- For any scripts use typer instead of argparse
- All the configurations comes fro

m config.yaml and iots parsed for better gitops approach and versioning
- The root contains src/, dockers, docker compos, make files, readme and all the code inside src, except for tests, docs, scripts etc.
- Always use clear naming for functions and classes and use PEP8 for naming
- Include minimial CI when implementing for testing and linting using github actions
- Try to use pattern desings when required and make sure it has an advantage, for example repository pattern, builder pattern, factory pattern, singleton (when instancing databases).
- For fastapi endpoint use injection dependecny for databases etc


## testing 
- For testing use pytest with covarage and other plugins that make sense (never overcomplicate, kiss principle)
- Use testcontainers if needed for infraestructure and real cases

Backend Skill Guidelines

Architecture
- Python with strict type hints
- Pydantic for validation
- FastAPI for APIs
- Light DDD (domain, application, infrastructure)
- Follow SOLID principles
- Prefer composition over inheritance
- Keep functions small and focused
- KISS principle

Code quality
- Use PEP8 naming
- Use Ruff and MyPy
- Avoid prints; use structured logging
- Clear naming for functions and classes
- Use design patterns only when they bring real benefit

Project structure
- src/ contains all application code
- tests/, docs/, scripts/ outside src
- config.yaml for configuration
- Docker and docker-compose at root
- Include minimal CI (GitHub Actions)

Security
- Validate all inputs
- Never expose secrets
- Use dependency scanning
- Avoid unsafe shell execution

Testing
- pytest with coverage
- pytest-asyncio
- testcontainers for infra tests
- Separate unit / integration tests

Observability
- structured logging
- request IDs
- proper log levels

API design
- versioned APIs
- clear FastAPI tags
- consistent error responses
- pagination for large lists

Use Python with strict type hints across the entire codebase so that every function, method, and public interface clearly specifies the expected input and output types, because strong typing greatly improves readability, prevents entire classes of runtime bugs, enables better static analysis with tools such as MyPy, and makes the system easier to maintain as the project grows.
Use Pydantic models for validation and data parsing so that all incoming data from APIs, configuration files, or external services is validated against explicit schemas, because unvalidated input is one of the most common sources of runtime errors and security vulnerabilities, and schema validation ensures that the system always operates on well-defined data structures.
Use FastAPI as the framework for building APIs because it provides automatic request validation, OpenAPI documentation generation, dependency injection, and asynchronous request handling out of the box, which helps maintain a clean architecture and significantly reduces boilerplate code while ensuring high performance and strong typing integration.
Organize the codebase using a light Domain Driven Design approach where business logic is separated into clear layers such as domain, application, and infrastructure, because this separation prevents business rules from being mixed with framework code, databases, or external services, making the system easier to test, evolve, and reason about over time.
Follow SOLID principles when designing classes and modules so that responsibilities remain clearly separated and the system remains flexible to change, because principles such as single responsibility and dependency inversion reduce tight coupling and ensure that components can evolve independently without breaking the entire application.
Prefer composition over inheritance when designing components so that behaviour is assembled through dependencies rather than rigid class hierarchies, because composition creates more flexible systems, improves testability, avoids deep inheritance chains, and makes it easier to replace or extend behaviour without modifying existing classes.
Keep functions small and focused on a single responsibility so that each function performs one clear task and remains easy to understand, because small well-defined functions reduce cognitive complexity, improve testability, and allow developers to reason about the behaviour of the system more effectively.
Apply the KISS principle (Keep It Simple, Stupid) when implementing solutions so that the architecture and code remain straightforward and avoid unnecessary abstraction layers, because overly complex systems become difficult to maintain, harder to debug, and slower to evolve.
Follow PEP8 naming and formatting conventions consistently so that the entire codebase maintains a uniform style that is easy for any developer to read and understand, because consistent formatting reduces friction when collaborating across teams and allows automated tools to enforce code quality.
Use modern static analysis and linting tools such as Ruff for code style enforcement and MyPy for type checking so that potential issues are detected early in development, because automated analysis prevents many categories of bugs and ensures that coding standards remain consistent across the entire project.
Avoid using print statements for debugging or runtime messages and instead rely on structured logging systems, because logging frameworks provide proper log levels, structured output, integration with monitoring systems, and the ability to filter or aggregate logs in production environments.
Use clear and descriptive naming for functions, classes, modules, and variables so that the purpose of each component is immediately understandable without requiring additional comments, because good naming is one of the most effective ways to make software self-documenting.
Use design patterns such as repositories, factories, builders, or strategies only when they bring clear architectural benefits, because unnecessary abstraction or premature pattern usage can make code harder to understand and maintain rather than improving it.
Place all application source code inside a dedicated src/ directory so that the project maintains a clean separation between executable code and other resources, because this structure improves import safety, avoids accidental imports from the project root, and keeps the codebase organized.
Keep directories such as tests, documentation, scripts, and tooling outside of the application source directory so that the runtime code remains isolated from development resources, which helps maintain a clean project layout and prevents accidental inclusion of non-production files in builds.
Store configuration in a structured configuration file such as config.yaml that can be versioned and managed through infrastructure workflows, because centralized configuration management makes the system easier to deploy, maintain, and audit in GitOps-based environments.
Place Dockerfiles and Docker Compose configurations at the root of the project so that the containerization setup is clearly visible and standardized across environments, because infrastructure configuration should be easily discoverable and consistent for developers and deployment pipelines.
Include a minimal continuous integration pipeline such as GitHub Actions that automatically runs tests, linting, and type checking on every commit, because automated validation ensures that code quality remains consistent and prevents broken code from entering the main branch.
Validate all external inputs including API payloads, query parameters, configuration values, and environment variables before processing them, because trusting unvalidated input can lead to runtime errors, inconsistent data states, and potential security vulnerabilities.
Never expose secrets such as API keys, database passwords, or authentication tokens in source code, configuration files committed to version control, or logs, because secrets must be handled through environment variables or dedicated secret management systems to prevent credential leakage.
Use dependency vulnerability scanning tools to regularly analyze Python dependencies and container images, because modern software relies heavily on third-party packages and known vulnerabilities in dependencies are one of the most common attack vectors.
Avoid executing shell commands directly from application code unless absolutely necessary and ensure that any command execution is carefully sanitized, because unsafe shell execution can lead to command injection vulnerabilities that allow attackers to execute arbitrary commands on the host system.
Use pytest as the primary testing framework with coverage reporting so that the reliability of the codebase can be measured and improved over time, because consistent automated testing helps detect regressions early and ensures that core functionality remains stable.
Use pytest-asyncio when testing asynchronous code so that asynchronous functions and event loops can be properly tested in environments where FastAPI or async services are used, because asynchronous behaviour introduces complexity that requires dedicated testing support.
Use testcontainers when infrastructure components such as databases, message brokers, or external services must be tested realistically, because running real containerized dependencies during tests provides higher confidence than using simple mocks.
Separate unit tests and integration tests clearly so that small isolated tests run quickly while broader system tests validate real component interactions, because separating test layers improves debugging speed and keeps the testing strategy scalable.
Use structured logging across the entire application so that logs contain consistent metadata such as timestamps, request identifiers, and contextual fields, because structured logs can be easily indexed and analyzed by observability platforms.
Include request identifiers in logs and request processing flows so that individual API requests can be traced across services, because request correlation is critical for debugging distributed systems and diagnosing production incidents.
Use proper logging levels such as DEBUG, INFO, WARNING, and ERROR consistently so that log verbosity can be controlled depending on the environment, because production environments require different logging behaviour than development environments.
Version APIs explicitly using version prefixes such as /v1 so that future changes to the API can be introduced without breaking existing clients, because stable API contracts are essential for long-lived services.
Use clear and well-organized FastAPI tags to group endpoints logically within API documentation, because structured API documentation improves discoverability for developers and consumers of the API.
Return consistent error responses from APIs with standardized structures so that clients can reliably interpret failures and handle them appropriately, because inconsistent error formats make integrations fragile and difficult to maintain.
Implement pagination for endpoints that return large collections of data so that responses remain efficient and scalable even as datasets grow, because unbounded responses can degrade performance, increase memory usage, and create unnecessary load on the server and clients.


## cibersecurity

Always use minimal base images and pin specific versions instead of using floating tags such as latest, because minimal images significantly reduce the attack surface by including fewer packages and system utilities that could contain vulnerabilities, and pinning versions ensures that builds remain reproducible over time and prevents unexpected updates from introducing security issues or breaking your environment.
Always run containers as a non-root user inside the container environment, because by default Docker runs processes as root which means that if an attacker exploits the application they will gain root privileges within the container, and although this is not automatically root on the host it greatly increases the risk of container escape, privilege escalation, or malicious modification of the container filesystem.
Use read-only filesystems for containers whenever possible so that the application cannot write arbitrary files to disk, because malware, cryptominers, and persistence mechanisms typically require writing files or installing binaries, and making the filesystem read-only significantly reduces the ability of an attacker to persist inside a compromised container.
Drop all unnecessary Linux capabilities from containers and explicitly enable only the ones required by the application, because containers inherit a set of kernel capabilities that grant powerful permissions such as network control or system administration features, and reducing these privileges follows the principle of least privilege which limits the damage that compromised software can cause.
Prevent privilege escalation inside containers by enabling security settings such as disabling the ability of processes to gain additional privileges during runtime, because many container attacks rely on exploiting binaries or system features that allow processes to elevate privileges, and preventing this behaviour ensures that the application cannot acquire permissions beyond those originally assigned.
Apply strict CPU and memory resource limits to all containers in order to protect the host system from denial-of-service situations, because compromised containers or poorly written applications may consume excessive resources such as memory or CPU which can destabilize the entire host machine and affect other services running on the same infrastructure.
Isolate services using separate Docker networks so that only the containers that truly need to communicate with each other share the same network, because flat networking between all services allows attackers who compromise one container to easily move laterally and access internal services such as databases, message queues, or internal APIs.
Avoid exposing container ports directly to the host unless absolutely necessary, because every exposed port increases the external attack surface and makes services reachable from outside the system, while internal services such as databases should remain accessible only through private container networks.
Never store secrets, API keys, database passwords, or tokens inside Docker images, configuration files committed to version control, or Dockerfiles, because once secrets are embedded in images they become extremely difficult to revoke and may leak through registries or logs, therefore secrets should always be injected at runtime through environment variables or dedicated secret management systems.
Regularly scan Docker images for known vulnerabilities before deployment, because container images include operating system libraries and third-party dependencies that may contain security flaws, and automated vulnerability scanners can detect outdated or insecure packages before they reach production environments.
Never mount sensitive host paths inside containers unless absolutely necessary, because mounting directories such as the host filesystem or critical system paths allows processes inside the container to read or modify files on the host machine, which can completely compromise the host if the container is exploited.
Avoid mounting the Docker socket (/var/run/docker.sock) inside containers because this effectively grants the container full control over the Docker daemon, allowing it to start new privileged containers, mount host filesystems, or escape the container environment entirely, which is equivalent to giving root access to the host system.
Restrict volumes to the minimum necessary directories and avoid sharing large portions of the host filesystem with containers, because volumes act as a bridge between the host and container environments and improperly configured mounts can allow attackers to access sensitive files or modify application data outside the intended boundaries.
Use multi-stage Docker builds so that development tools, compilers, and build dependencies are not included in the final runtime image, because build tools increase the attack surface and may provide attackers with utilities that make exploitation easier if they gain access to the container.
Implement health checks for containers to detect when services become unhealthy, compromised, or stuck in faulty states, because automated health monitoring allows orchestration systems to restart or replace failing containers and reduces the risk of long-running compromised services remaining active.
Use structured logging instead of simple print statements and avoid logging sensitive information such as credentials or tokens, because logs often end up in centralized logging systems and exposing secrets in logs can lead to serious security incidents if those logs are accessed by unauthorized users.
Ensure file permissions inside containers follow the principle of least privilege so that application directories are writable only when necessary and executable permissions are not broadly granted, because overly permissive filesystem permissions allow malicious processes to alter application files or inject additional code.
Continuously audit application dependencies and Python packages for vulnerabilities using automated tools, because modern software relies heavily on third-party libraries and supply chain attacks or outdated dependencies are one of the most common sources of security breaches in containerized environments.
Use a reverse proxy or API gateway in front of application containers rather than exposing backend services directly to the internet, because reverse proxies can enforce TLS, rate limiting, request filtering, and authentication policies which greatly reduce the attack surface of the application layer.
Regularly update container images, operating system packages, and dependencies to ensure that known vulnerabilities are patched promptly, because attackers frequently target publicly known security flaws in outdated software components and maintaining updated environments significantly reduces this risk.
Design container environments under the assumption that a container might eventually be compromised and therefore ensure that strong isolation exists between services, because good container security architecture focuses not only on preventing attacks but also on limiting the impact when an attack successfully occurs.
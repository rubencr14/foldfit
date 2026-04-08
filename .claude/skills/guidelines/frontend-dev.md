En frontend eso acaba generando ceremonias, factories inútiles y pseudo-OO. Lo que mejor funciona en Next/React suele ser una mezcla de:

Screaming Architecture / feature-first
Ports & Adapters / Clean Architecture ligera
Presentational + Container
Composition over inheritance
Dependency injection simple
State management mínimo y local por defecto
Contrato fuerte con TypeScript
Testing pyramid bien aterrizada
El patrón base que yo usaría
1. Organiza por feature, no por tipo global

Evita esto:

/components
/hooks
/services
/utils
/types

porque al cabo de meses se convierte en un cajón desastre.

Prefiere algo así:

src/
  app/
    (marketing)/
    dashboard/
      recipes/
        page.tsx
        loading.tsx
        error.tsx

  features/
    recipes/
      components/
      hooks/
      application/
      domain/
      infrastructure/
      tests/

    auth/
    billing/

  shared/
    components/
    lib/
    types/
    test/

Eso hace que cada feature tenga su propia mini-arquitectura y no acabes con un monolito de frontend.

2. Dentro de cada feature: 4 capas

Yo usaría esta separación:

features/recipes/
  domain/
    recipe.ts
    recipe.repository.ts
    recipe.errors.ts

  application/
    get-recipes.use-case.ts
    save-recipe.use-case.ts

  infrastructure/
    recipe-api.repository.ts
    recipe-dto.ts
    recipe-mappers.ts

  components/
    RecipeList.tsx
    RecipeCard.tsx
    RecipePageClient.tsx
Qué va en cada una

domain/

entidades
value objects
reglas de negocio puras
interfaces de repositorio

application/

casos de uso
orquestación
llama a puertos/interfaces
sin JSX, sin fetch directo si puedes evitarlo

infrastructure/

implementación real
fetch, axios, SDKs, localStorage, cookies, APIs
mappers DTO ↔ dominio

components/

UI pura
recibe datos ya listos
lógica visual, no de negocio

Este patrón te da desacoplamiento real.

3. Ports & Adapters en frontend sí, pero ligero

Como ya prefieres “arquitectura de puertos y adaptadores”, aquí encaja muchísimo.

Ejemplo:

// domain/recipe.repository.ts
export interface RecipeRepository {
  list(): Promise<Recipe[]>;
  getById(id: string): Promise<Recipe | null>;
  save(recipe: Recipe): Promise<void>;
}
// application/get-recipes.use-case.ts
import type { RecipeRepository } from "../domain/recipe.repository";

export class GetRecipesUseCase {
  constructor(private readonly recipeRepository: RecipeRepository) {}

  async execute() {
    return this.recipeRepository.list();
  }
}
// infrastructure/recipe-api.repository.ts
import type { RecipeRepository } from "../domain/recipe.repository";

export class RecipeApiRepository implements RecipeRepository {
  async list() {
    const res = await fetch("/api/recipes");
    const data = await res.json();
    return data.items;
  }

  async getById(id: string) {
    const res = await fetch(`/api/recipes/${id}`);
    if (res.status === 404) return null;
    return res.json();
  }

  async save(recipe) {
    await fetch("/api/recipes", {
      method: "POST",
      body: JSON.stringify(recipe),
    });
  }
}

Beneficio: mañana cambias REST por GraphQL, IndexedDB o mocks y no tocas el caso de uso.

4. Presentational vs Container sigue funcionando muy bien

En React moderno no hace falta llamarlo siempre así, pero conceptualmente sigue siendo oro.

Malo

Un componente que:

hace fetch
transforma DTOs
valida permisos
decide navegación
renderiza UI
Mejor
Container / page / hook: obtiene y prepara datos
Presentational: solo pinta
// components/RecipeList.tsx
type Props = {
  recipes: RecipeViewModel[];
  onSelect: (id: string) => void;
};

export function RecipeList({ recipes, onSelect }: Props) {
  return (
    <ul>
      {recipes.map((recipe) => (
        <li key={recipe.id}>
          <button onClick={() => onSelect(recipe.id)}>
            {recipe.title}
          </button>
        </li>
      ))}
    </ul>
  );
}

Este componente es facilísimo de testear.

React insiste mucho en pensar en componentes, props y flujo de datos claro, y usar composición como base.

5. Server Components por defecto, Client Components solo donde haga falta

En Next.js App Router, este patrón es clave.

Usa Server Components para:

fetch inicial
composición de página
auth del lado servidor
acceso a backend
datos SEO y metadata

Usa Client Components solo para:

interacción
formularios ricos
estado local UI
drag & drop
modales
autocompletado, etc.

Eso reduce JS en cliente y mantiene la lógica más limpia. Next.js recomienda App Router y su modelo de Server/Client Components para proyectos nuevos.

Regla práctica
page.tsx: composición y fetch del lado servidor
SomethingClient.tsx: interactividad
los casos de uso no viven dentro del componente
6. Custom hooks sí, pero solo para lógica de UI o coordinación

Muchos equipos meten toda la app en hooks gigantes. Error clásico.

Usa hooks para:

estado local reutilizable
sincronización con router
debounce
formularios
coordinación entre UI y caso de uso

No los uses como “capa mágica donde esconder todo”.

Bien
export function useRecipeSearch() {
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 300);

  return { query, setQuery, debouncedQuery };
}
Mal

Un useRecipes() de 800 líneas con fetch, auth, analytics, cache, redirects y toasts.

7. View Models / Mappers entre backend y UI

Muy importante para evitar acoplar la UI al JSON del backend.

Nunca hagas esto
<p>{recipe.owner.first_name}</p>

si eso viene directo del backend y mañana cambia.

Haz esto:

export type RecipeCardViewModel = {
  id: string;
  title: string;
  authorName: string;
  imageUrl: string | null;
};
export function toRecipeCardViewModel(dto: RecipeDto): RecipeCardViewModel {
  return {
    id: dto.id,
    title: dto.title,
    authorName: `${dto.owner.first_name} ${dto.owner.last_name}`,
    imageUrl: dto.main_image_url ?? null,
  };
}

Beneficio: la UI consume contratos estables y legibles.

8. Tipado fuerte: interfaces y types como contratos, no como decoración

TypeScript sigue recomendando usar tipos de objeto e interfaces como contratos estructurales.

Mi regla práctica:

interface para contratos extendibles y puertos
type para unions, view models, utilidades y composición de tipos
Ejemplo
export interface RecipeRepository {
  list(): Promise<Recipe[]>;
}
export type RecipeStatus = "draft" | "published" | "archived";

No crees tipos por crear. Cada tipo debe representar una frontera o una intención real.

9. Estado: local primero, global cuando duela de verdad

Orden recomendado:

estado local del componente
props
context para cross-cutting pequeño
server state con fetch/cache
estado global solo si realmente compartes mucha interacción

React dice claramente que uses props como vía principal y context cuando pasar props se vuelve incómodo en árboles profundos.

Consejo duro pero útil

No metas Zustand/Redux/Context global para todo al principio.
Primero diseña bien las fronteras. Muchas veces el “problema de estado global” era un problema de arquitectura.

10. Inversión de dependencias simple

No hace falta un contenedor IoC enorme.

Puedes inyectar dependencias con funciones o factories pequeñas:

type Dependencies = {
  recipeRepository: RecipeRepository;
};

export function makeGetRecipesUseCase({ recipeRepository }: Dependencies) {
  return {
    execute: () => recipeRepository.list(),
  };
}

Y en Next:

const recipeRepository = new RecipeApiRepository();
const getRecipes = makeGetRecipesUseCase({ recipeRepository });
const recipes = await getRecipes.execute();

Esto hace testing muy fácil con fakes.

11. Validación en los bordes

Toda entrada externa debería validarse:

request params
forms
backend responses
env vars

Aunque aquí no he buscado librerías concretas, el patrón importante es:
valida en el borde, usa tipos estables dentro.

12. Testing por capas

Aquí es donde muchas arquitecturas bonitas fallan.

Next documenta Vitest/Jest para unit tests y Playwright para E2E. También advierte que los async Server Components encajan mejor con E2E que con unit testing puro.

Yo haría esto:

A. Unit tests

Para:

domain
application
mappers
componentes presentacionales puros

Ejemplos:

toRecipeCardViewModel
GetRecipesUseCase
RecipeCard
B. Integration tests

Para:

repositorios
formularios conectados
páginas con providers
API client + mocks
C. E2E

Para:

login
flujos críticos
guardado de receta
checkout
búsqueda principal
Regla

No intentes testear todo desde UI.
Y no intentes testear Server Components async como si fueran funciones puras del todo.

13. Documentación: docstrings sí, pero donde aporten valor

No llenaría el frontend de comentarios obvios.

Sí pondría docstrings en:
casos de uso
contratos
funciones con reglas de negocio
mappers no triviales
hooks reutilizables
componentes de librería interna

Ejemplo bueno:

/**
 * Returns recipes visible to the current user, applying publication
 * and ownership rules. This use case must not depend on React or Next.js.
 */
export class GetVisibleRecipesUseCase {
  // ...
}
No pondría esto
// increment count by 1
setCount(count + 1);

La mejor documentación en frontend es:

nombres buenos
módulos pequeños
contratos claros
README por feature cuando la lógica lo merezca
14. Componentes pequeños, pero no absurdamente fragmentados

No caigas en:

ButtonLabel.tsx
CardHeaderTitleWrapper.tsx

Un buen componente suele tener:

una responsabilidad visual clara
props claras
poca lógica incidental

Piensa en unidad de cambio.
Si dos trozos siempre cambian juntos, quizá deben vivir juntos.

15. Evita helpers genéricos sin dueño

Muy típico en monolitos frontend:

utils.ts
helpers.ts
common.ts
misc.ts

Eso es cementerio de arquitectura.

Mejor:

recipe-title.formatter.ts
auth-session.mapper.ts
price-display.ts

Todo con contexto.

16. Errores y resultados explícitos

No ocultes todo en throw new Error("Oops").

Puedes usar:

errores de dominio
resultados tipados
estados explícitos
type SaveRecipeResult =
  | { ok: true; id: string }
  | { ok: false; reason: "validation" | "forbidden" | "network" };

Esto mejora testabilidad y UI predecible.

17. Accesibilidad y UX como parte de la arquitectura

No es “detalle final”.
Si desde el principio tus componentes exponen bien:

aria-*
labels
roles
teclado

tus tests también mejoran, porque Testing Library trabaja mejor con UI accesible.

18. La arquitectura que yo te recomendaría de verdad para un SaaS serio en Next.js
src/
  app/
    dashboard/
      recipes/
        page.tsx
        loading.tsx
        error.tsx
        [id]/
          page.tsx

  features/
    recipes/
      domain/
        recipe.ts
        recipe.repository.ts
        recipe.errors.ts

      application/
        get-recipes.use-case.ts
        get-recipe-by-id.use-case.ts
        save-recipe.use-case.ts

      infrastructure/
        recipe-api.repository.ts
        recipe-dto.ts
        recipe-mappers.ts

      components/
        RecipeCard.tsx
        RecipeList.tsx
        RecipeEditorClient.tsx

      hooks/
        useRecipeFilters.ts

      tests/
        get-recipes.use-case.test.ts
        recipe-mappers.test.ts
        RecipeCard.test.tsx

  shared/
    components/
      ui/
    lib/
      env.ts
      fetcher.ts
    test/
      builders/
      mocks/
Mis “design patterns” favoritos para frontend real

Si me obligas a elegir los más útiles:

Composition over inheritance
React va por aquí claramente.
Ports & Adapters
Para desacoplar backend, storage, SDKs y servicios externos.
Presentational / Container
Para testabilidad y claridad.
Factory simple
Para construir casos de uso y dependencias.
Mapper / Adapter
Para no contaminar la UI con DTOs.
Strategy
Cuando cambian comportamientos por feature flag, rol, proveedor o modo.
State reducer pattern
Para estados complejos de formularios o editor.

No usaría alegremente:

Singleton por todos lados
clases de servicios gigantes
herencia
repos de “utils” globales
abstract factories ceremoniosas
Las reglas de oro que más evitan monolitos

Si tuviera que resumírtelo en reglas duras:

cada feature tiene dueño y carpeta propia
la UI no conoce el backend crudo
los casos de uso no conocen React
los repositorios implementan interfaces
los componentes reciben props simples
Client Components solo cuando hay interacción
tests unitarios para lógica, E2E para flujos
tipos como contratos en fronteras
mappers entre capas
nada de utils.ts genérico
Mi recomendación final

Para un proyecto tuyo en Next.js, sobre todo viendo que tiendes a productos grandes y con vida larga, usaría este stack conceptual:

Feature-first + App Router + Server Components por defecto + Clean Architecture ligera + puertos y adaptadores + mappers + Vitest/RTL + Playwright

Eso te da un frontend:

desacoplado
muy testeable
fácil de escalar
mucho menos monolítico

Y además encaja muy bien con tu forma de pensar en backend y sistemas.

Puedo prepararte ahora una plantilla real de estructura para Next.js 15+ con ejemplos de carpetas, interfaces, casos de uso, repositorios y tests
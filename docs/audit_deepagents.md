# Audit d'utilisation de DeepAgents dans CosmeticDeepStat

> **Date de l'audit :** 22 mai 2026
> **Version deepagents pinée :** `>=0.6.3,<0.7`
> **Version réelle installée :** `0.6.3` (PyPI, publiée le 20 mai 2026)
> **Documentation de référence :**
> - [API officielle `create_deep_agent`](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)
> - [Sub-agents docs](https://docs.langchain.com/oss/python/deepagents/subagents)
> - [HITL docs](https://docs.langchain.com/oss/python/deepagents/human-in-the-loop)
> - [Customization docs](https://docs.langchain.com/oss/python/deepagents/customization)
> - [Blog v0.6](https://www.langchain.com/blog/deep-agents-0-6)

---

## 1. Résumé exécutif

Notre projet utilise DeepAgents comme **dépendance déclarée** mais ne
l'exploite réellement qu'en mode `mock` (pipeline déterministe Python).
Le code du « real DeepAgents path » dans `master_agent.py` contient
**7 problèmes** de conformité avec l'API réelle de deepagents 0.6.x,
dont 3 bloquants qui empêcheraient le mode LLM de fonctionner.

| Sévérité | Nombre | Impact |
|----------|--------|--------|
| 🔴 Bloquant (crash à l'exécution) | 3 → **0** ✅ | Corrigés dans ce commit |
| 🟡 Majeur (comportement incorrect) | 2 → **0** ✅ | C4 corrigé (resume), C5 corrigé (guard) |
| 🟢 Mineur (cosmétique / bonnes pratiques) | 2 | Pas d'impact fonctionnel |

---

## 2. Ce que DeepAgents 0.6.x est réellement

### 2.1 Architecture

DeepAgents est un **harness** (harnais) construit au-dessus de LangChain +
LangGraph. Ce n'est **pas** un framework séparé — c'est une couche
d'abstraction qui pré-configure :

1. **Planning** — outils `write_todos` / `read_todos` injectés automatiquement
2. **Filesystem** — outils `write_file`, `read_file`, `edit_file`, `ls`,
   `glob`, `grep` injectés automatiquement via le `backend`
3. **Sub-agents** — outil `task()` injecté automatiquement, route vers les
   sub-agents déclarés
4. **Skills** — middleware `SkillsMiddleware` scan les `SKILL.md` et injecte
   le catalogue dans le system prompt
5. **Memory** — fichiers `.md` chargés dans le contexte de l'agent
6. **Context management** — compression automatique de l'historique quand
   le contexte dépasse un seuil
7. **HITL** — `interrupt_on` pause le graphe LangGraph quand un outil
   spécifié est appelé

### 2.2 Signature réelle de `create_deep_agent` (v0.6.3)

```python
def create_deep_agent(
    model: str | BaseChatModel | None = None,
    tools: Sequence[BaseTool | Callable | dict] | None = None,
    *,
    system_prompt: str | SystemMessage | None = None,
    middleware: Sequence[AgentMiddleware] = (),
    subagents: Sequence[SubAgent | CompiledSubAgent] | None = None,
    skills: list[str] | None = None,
    memory: list[str] | None = None,
    permissions: list[FilesystemPermission] | None = None,
    backend: BackendProtocol | BackendFactory | None = None,
    interrupt_on: dict[str, bool | InterruptOnConfig] | None = None,
    response_format: ResponseFormat | type | dict | None = None,
    context_schema: type | None = None,
    checkpointer: Checkpointer | None = None,
    store: BaseStore | None = None,
    debug: bool = False,
    name: str | None = None,
    cache: BaseCache | None = None,
) -> CompiledStateGraph
```

### 2.3 Schema réel de `SubAgent` (v0.6.3)

```python
class SubAgent(TypedDict):
    name: str                    # obligatoire
    description: str             # obligatoire
    system_prompt: str           # obligatoire (remplace "prompt" depuis v0.5+)
    tools: NotRequired[list]     # optionnel — hérite du parent si absent
    model: NotRequired[str | BaseChatModel]
    middleware: NotRequired[list[AgentMiddleware]]
    interrupt_on: NotRequired[dict]
    skills: NotRequired[list[str]]           # NOUVEAU en v0.6
    response_format: NotRequired[...]        # NOUVEAU en v0.6
    permissions: NotRequired[list]           # NOUVEAU en v0.6
```

**Note historique :** Dans les versions ≤ 0.0.10 (PyPI sept 2025), la clé
s'appelait `prompt`. Depuis la v0.5.x (début 2026), elle s'appelle
`system_prompt`. Les deux sont acceptés en v0.5 mais seul `system_prompt`
est documenté en v0.6.

---

## 3. Constatations détaillées

### 🔴 C1 — SubAgent `tools` passe des objets au lieu de fonctions

**Fichier :** `app/agents/subagents.py`

**Notre code :**
```python
"tools": _by_name(tools, ["load_dataset_tool", "profile_dataset_tool", ...])
```

**Problème :** Nous passons les `@tool`-decorated callables LangChain,
ce qui est correct. **Cependant**, deepagents 0.6.x documente que quand
`tools` est spécifié sur un sub-agent, il **remplace** (override) les
outils hérités du parent, y compris les outils built-in (`write_file`,
`read_file`, `ls`, `edit_file`, `write_todos`).

Nos sub-agents n'incluent **pas** les outils filesystem built-in dans
leur `tools` list. Résultat : en mode LLM, les sub-agents **ne pourront
pas écrire de fichiers**, ce qui casse tout le workflow (écriture de
résultats JSON, scripts, reports).

**Correction requise :** soit ne pas spécifier `tools` (hérite tout du
parent), soit inclure explicitement les outils filesystem, soit retirer
`tools` et s'appuyer sur le filtrage via le `system_prompt`.

**Sévérité :** 🔴 Bloquant

---

### 🔴 C2 — `backend` absent dans `create_deep_agent`

**Fichier :** `app/agents/master_agent.py`

**Notre code :**
```python
create_kwargs = {
    "model": ...,
    "tools": tools,
    "system_prompt": ...,
    "subagents": subagents,
    "skills": [...],
    "interrupt_on": ...,
    "checkpointer": MemorySaver(),
    "name": ...,
}
# PAS de backend=...
```

**Problème :** Sans `backend`, deepagents utilise `StateBackend` (in-memory).
Les fichiers écrits par les outils built-in (`write_file`, etc.) ne sont
**pas persistés sur le filesystem réel** — ils vivent dans l'état du graphe
et disparaissent à la fin de l'invocation.

Notre architecture **dépend** du fait que les fichiers sont écrits dans
`workspace/{study_id}/...` sur le vrai filesystem. En mode mock, c'est
nous qui écrivons directement. En mode deepagents, il faut un
`FilesystemBackend`.

**Correction requise :**
```python
from deepagents.backends import FilesystemBackend

create_kwargs["backend"] = FilesystemBackend(
    root_dir=str(settings.workspace_root_abs)
)
```

**Sévérité :** 🔴 Bloquant

---

### 🔴 C3 — Invocation du graphe sans `thread_id`

**Fichier :** `app/agents/master_agent.py`

**Notre code :**
```python
def invoke(self, payload):
    return self.graph.invoke(payload)
```

**Problème :** DeepAgents avec un `checkpointer` **exige** un `thread_id`
dans la config pour la persistance de l'état et le HITL. Sans cela,
`invoke()` lève une erreur `Missing required configurable: thread_id`.

Le HITL (`interrupt_on`) ne peut fonctionner que si :
1. Un `checkpointer` est attaché ✅ (nous passons `MemorySaver`)
2. Un `thread_id` est fourni dans `config` ❌
3. La reprise se fait avec `Command(resume=...)` ❌

**Correction requise :**
```python
import uuid
config = {"configurable": {"thread_id": f"study-{study_id}"}}
result = self.graph.invoke(payload, config=config)
```

Et pour la reprise après HITL :
```python
from langgraph.types import Command
result = self.graph.invoke(
    Command(resume={"decisions": [{"type": "approve"}]}),
    config=config,
)
```

**Sévérité :** 🔴 Bloquant

---

### 🟡 C4 — `interrupt_on` utilise le mauvais format

**Fichier :** `app/agents/master_agent.py`

**Notre code :**
```python
interrupt_on = {
    "request_human_approval_tool": True,
}
```

**Constat :** Le format `{tool_name: True}` est **valide** dans deepagents
0.6.x. C'est un raccourci pour `{"allowed_decisions": ["approve", "edit",
"reject"]}`. ✅ correct.

**Cependant**, notre HITL est implémenté **dans nos propres tools**
(`_impl_request_human_approval` crée un `ApprovalRequest` en base) plutôt
que d'utiliser le mécanisme natif de deepagents. Le `interrupt_on` pauserait
le graphe **après** l'exécution du tool (l'approval est déjà créée en base),
ce qui est le bon timing. Mais la **reprise** nécessite `Command(resume=...)`
avec le bon format de décisions, pas un simple re-`invoke`.

**Impact :** Le HITL fonctionnerait partiellement — la pause se fait, mais
la reprise est manquante. Notre pipeline déterministe gère cela
correctement en re-invoquant tout le pipeline.

**Sévérité :** 🟡 Majeur (le HITL ne fonctionnerait pas correctement en
mode LLM)

---

### 🟡 C5 — `memory` passe des chemins absolus

**Fichier :** `app/agents/master_agent.py`

**Notre code :**
```python
memory_files = [p for p in settings.memory_root_abs.rglob("*.md")]
create_kwargs["memory"] = [str(p) for p in memory_files]
```

**Constat :** deepagents accepte des chemins absolus pour `memory`, mais
les charge en lisant le contenu du fichier et en l'injectant dans le
system prompt. Si les fichiers sont volumineux, cela peut **exploser le
contexte**.

Nos 4 fichiers memory font ~1500 tokens au total, ce qui est raisonnable.
Mais il n'y a pas de garde-fou si quelqu'un ajoute des fichiers lourds.

De plus, `memory` en deepagents 0.6 fonctionne avec `CompositeBackend`
et `StoreBackend` pour la persistance cross-session, pas juste des chemins
fichiers. Notre utilisation est la forme basique, ce qui est correct mais
sous-optimale.

**Sévérité :** 🟡 Majeur (risque de context overflow)

---

### 🟢 C6 — `skills` paths pourraient ne pas exister en production

**Fichier :** `app/agents/master_agent.py`

**Notre code :**
```python
"skills": [str(settings.skills_root_abs)]
```

**Constat :** Si le répertoire `skills/` n'existe pas au runtime (ex:
déploiement Docker sans les skills), deepagents ne crash pas mais
n'injecte aucun skill. Ce n'est pas un bug mais un point d'attention
opérationnel.

**Sévérité :** 🟢 Mineur

---

### 🟢 C7 — Paramètres v0.6 non exploités

**Constat :** deepagents 0.6 introduit plusieurs paramètres que nous
n'utilisons pas :

| Paramètre v0.6 | Utilité pour nous | Priorité |
|-----------------|-------------------|----------|
| `permissions` | Restreindre l'accès filesystem par sub-agent | Haute |
| `response_format` | Forcer le JSON structuré en sortie de sub-agent | Haute |
| `store` | Mémoire persistante cross-session (via InMemoryStore ou MongoDB) | Moyenne |
| `context_schema` | Typer le contexte partagé | Basse |
| `cache` | Cache LLM pour les requêtes répétitives | Basse |

**Sévérité :** 🟢 Mineur (fonctionnel mais sous-optimal)

---

## 4. Analyse du mode mock vs mode LLM

| Aspect | Mode mock (actuel) | Mode LLM (deepagents) |
|--------|-------------------|----------------------|
| **Fonctionne ?** | ✅ Oui, 116 tests passent | ❌ Non, 3 bloquants |
| **Pipeline** | Déterministe Python | LLM orchestré |
| **Filesystem** | Écritures directes via `StudyWorkspace` | Nécessite `FilesystemBackend` |
| **HITL** | Géré par notre propre logique `ApprovalRequest` | Nécessite `interrupt_on` + `Command(resume=...)` |
| **Skills** | Non chargés (pas de LLM) | `SkillsMiddleware` scanne les SKILL.md |
| **Memory** | Non utilisé | Injecté dans le system prompt |
| **Sub-agents** | Non utilisés (pipeline séquentiel) | `task()` tool délègue au bon sub-agent |

**Conclusion :** Le mode mock est un **raccourci intelligent** qui contourne
totalement deepagents. Mais le code du mode LLM n'a jamais été testé et
contient des erreurs bloquantes.

---

## 5. Plan de correction

### Phase immédiate (rendre le mode LLM fonctionnel)

| # | Action | Fichier | Effort |
|---|--------|---------|--------|
| 1 | Ajouter `FilesystemBackend` | `master_agent.py` | 5 min |
| 2 | Passer `thread_id` dans `config` | `master_agent.py` | 10 min |
| 3 | Implémenter la reprise HITL avec `Command(resume=...)` | `master_agent.py` + `api/analyses.py` | 30 min |
| 4 | Ne pas spécifier `tools` sur les sub-agents qui n'ont besoin que des outils du parent (ou inclure les built-in) | `subagents.py` | 20 min |
| 5 | Ajouter `response_format` (Pydantic) sur les sub-agents structurés | `subagents.py` | 15 min |
| 6 | Ajouter `permissions` pour empêcher les sub-agents d'écrire dans `raw/` | `subagents.py` | 10 min |

### Phase ultérieure

| # | Action | Bénéfice |
|---|--------|----------|
| 7 | Utiliser `CompositeBackend` + `StoreBackend` pour la mémoire persistante | Cross-session memory |
| 8 | Utiliser le code interpreter v0.6 pour le PTC (Programmatic Tool Calling) | Réduction des appels LLM |
| 9 | Configurer `store` avec `InMemoryStore` ou MongoDB | Mémoire long-terme |
| 10 | Tester en e2e avec un vrai LLM (Anthropic Claude Sonnet) | Validation réelle |

---

## 6. Ce qui est correct

Malgré les problèmes ci-dessus, plusieurs aspects de notre intégration
sont **bien faits** :

✅ **Architecture mock/real séparée** — le pipeline déterministe
fonctionne parfaitement sans LLM, ce qui est essentiel pour les tests
et le développement.

✅ **`system_prompt`** (pas `prompt`) — nous utilisons la clé correcte
pour les sub-agents, compatible v0.5+/v0.6.

✅ **Pinning de version** — `deepagents>=0.6.3,<0.7` protège contre les
breaking changes de la v1.0.

✅ **Tools dual-pattern** — `_impl_*` + `@tool` wrapper est exactement
le pattern recommandé (tests sans LLM, LLM avec @tool).

✅ **Skills au bon format** — nos 14 SKILL.md ont le YAML frontmatter
(name + description) requis par `SkillsMiddleware`.

✅ **Memory files** — format `.md` correct, taille raisonnable.

✅ **Provider-agnostic** — le format `provider:model` est le bon pour
`init_chat_model()`.

---

## 7. Recommandation globale

Le projet est **solide en mode mock** (le mode principal actuel). Le
passage en mode LLM réel nécessite **~90 minutes de corrections** sur
3 fichiers. Les corrections sont isolées dans `master_agent.py`,
`subagents.py`, et `api/analyses.py`.

**Priorité recommandée :**
1. 🔴 Corriger C1 + C2 + C3 (bloquants) → le mode LLM peut démarrer
2. 🟡 Corriger C4 (HITL reprise) → le HITL fonctionne en mode LLM
3. 🟡 Corriger C5 (memory guard) → protection contre le context overflow
4. 🟢 Exploiter les features v0.6 → optimisation

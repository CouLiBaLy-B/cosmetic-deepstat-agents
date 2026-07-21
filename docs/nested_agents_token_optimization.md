# Deep Agents imbriqués & optimisation des tokens en input

> **Référence :** deepagents 0.6.x (vérifié contre `deepagents 0.6.12`, LangChain OSS, juillet 2026)
> **Objectif :** un agent maître dont les sous-agents sont eux-mêmes des *deep agents*
> qui ont leurs propres sous-agents, afin de **minimiser le nombre de tokens en input**.

---

## 1. Le problème (topologie « plate »)

Dans la topologie initiale, `create_deep_agent` du master reçoit :

```python
create_kwargs = {
    "tools": tools,          # ← les 19 outils custom
    "subagents": subagents,  # ← les 10 sous-agents (name + description)
    ...
}
```

Conséquence : **à chaque tour LLM du master**, le contexte d'entrée transporte :

- les **schémas JSON des 19 outils** (nom + description + signature des arguments) ;
- les **10 descriptions de sous-agents** injectées dans l'outil `task`.

La quasi-totalité est **inutile à l'étape courante** (quand le master mappe des
claims, il n'a pas besoin du schéma de `run_mmrm_tool`). Ce coût est payé
**à chaque tour**, donc il se cumule sur toute l'orchestration.

Mesure sur ce dépôt (approximation 4 char/token, surface de délégation du master) :

| Topologie | Outils | Sous-agents/équipes | **Total master / tour** |
|-----------|:------:|:-------------------:|:-----------------------:|
| Plate     | ~1962 tok (19 outils) | ~522 tok (10 desc) | **~2484 tok** |
| Imbriquée | 0 tok (0 outil custom) | ~218 tok (4 équipes) | **~218 tok** |

➡️ **~91 % de réduction** de la surface de délégation portée par le master à chaque tour.

---

## 2. La solution deepagents 2026 : `CompiledSubAgent`

deepagents 0.6 accepte trois formes de sous-agents :

```python
subagents: Sequence[SubAgent | CompiledSubAgent | AsyncSubAgent]
```

- **`SubAgent`** — un dict déclaratif `{name, description, system_prompt, tools?, ...}`.
- **`CompiledSubAgent`** — enveloppe **n'importe quel graphe LangGraph compilé**,
  y compris **un autre `create_deep_agent(...)`**. C'est le mécanisme officiel
  d'imbrication de deep agents.
- **`AsyncSubAgent`** — sous-agent distant.

```python
from deepagents import create_deep_agent, CompiledSubAgent

# 1. Un deep agent spécialisé (avec SES propres outils + SES propres sous-agents)
team_graph = create_deep_agent(
    model=model,
    tools=team_tools,            # sous-ensemble étroit
    system_prompt=TEAM_PROMPT,
    subagents=team_members,      # les spécialistes de l'équipe
    name="evidence_team",
)

# 2. On l'expose au master comme UN seul sous-agent
nested = CompiledSubAgent(
    name="evidence_team",
    description="Data quality + stats + consumer insight.",
    runnable=team_graph,         # doit être .compile() (create_deep_agent l'est déjà)
)

master = create_deep_agent(model=model, subagents=[nested, ...])
```

> ⚠️ En 0.6.x, `CompiledSubAgent` est un **TypedDict** : `CompiledSubAgent(...)`
> renvoie un `dict` `{name, description, runnable}`. Le champ `runnable` doit
> exposer une clé d'état `messages` (c'est le cas d'un deep agent).
> À la fin du sous-agent, **seul le dernier message (résumé)** est renvoyé au
> parent sous forme de `ToolMessage` — les appels d'outils intermédiaires ne
> remontent **jamais** dans le contexte du parent.

---

## 3. Architecture à 3 niveaux appliquée au projet

```text
                    ┌───────────────────────────────────────┐
 Niveau 0 (master)  │  cosmetic_evidence_orchestrator        │
                    │  outils custom : 0                     │
                    │  voit : 4 descriptions d'équipe        │
                    └───┬───────────┬───────────┬───────────┬┘
                        │ task()    │           │           │
        ┌───────────────▼──┐  ┌─────▼───────┐  ┌▼──────────────┐  ┌▼────────────┐
Niveau 1│ protocol_team    │  │ evidence_    │  │ decision_     │  │ reporting_  │
(chefs  │ deep agent       │  │ team         │  │ safety_team   │  │ team        │
 deep   │ outils: 3        │  │ outils: 16   │  │ outils: 7     │  │ outils: 3   │
 agents)└──┬────────────┬──┘  └─┬────┬────┬──┘  └─┬────┬────┬───┘  └──┬───────┬──┘
           │ task()     │       │    │    │       │    │    │         │       │
Niveau 2 ┌─▼──┐      ┌──▼─┐  ┌──▼┐ ┌─▼┐ ┌▼─┐   ┌─▼┐ ┌─▼┐ ┌▼─┐    ┌──▼┐   ┌──▼┐
(spécia- │reg.│      │stud│  │dq │ │st│ │co│   │mu│ │sa│ │pm│    │rep│   │qa │
 listes) │map │      │des.│  │   │ │at│ │ns│   │lt│ │fe│ │mon│   │wrt│   │aud│
         └────┘      └────┘  └───┘ └──┘ └──┘   └──┘ └──┘ └──┘    └───┘   └───┘
```

### Partition des 10 spécialistes en 4 équipes

| Équipe (Niv. 1) | Spécialistes (Niv. 2) | Outils custom | Gate HITL |
|-----------------|-----------------------|:-------------:|-----------|
| `protocol_team` | regulatory_claim_mapper, study_design_subagent | 3 | SAP |
| `evidence_team` | data_quality, statistical_analysis, consumer_insight | 16 | — |
| `decision_safety_team` | multiplicity_claim, safety_tolerability, postmarket_monitoring | 7 | claims, safety |
| `reporting_team` | report_writer, qa_auditor | 3 | rapport final |

`tool_names` de chaque équipe = **union** des outils de ses membres (jamais un
outil qu'aucun membre ne peut utiliser — vérifié par test).

---

## 4. Les leviers d'optimisation de tokens (cumulés)

deepagents combine plusieurs mécanismes ; nous les exploitons tous :

1. **Isolation de contexte par hiérarchie** *(levier principal, ce dépôt)*
   Le master ne porte plus 19 outils + 10 desc, mais 0 outil + 4 desc. Chaque
   chef d'équipe ne porte que son sous-ensemble étroit d'outils + 2-3 membres.
   → `agent_topology="nested"` (par défaut).

2. **Contexte frais par sous-agent + résumé remontant**
   Chaque `task()` s'exécute dans une fenêtre de contexte isolée ; seul le
   résumé final revient. Les profils de datasets, dumps stats, etc. ne polluent
   jamais le parent.

3. **Offloading automatique sur le filesystem** (> 20 000 tokens)
   deepagents remplace automatiquement les entrées/sorties d'outils volumineuses
   par une référence fichier + un aperçu. Renforcé ici par la règle « jamais de
   dataset brut dans le prompt » + `FilesystemBackend`.

4. **`SummarizationMiddleware`** (compression automatique ~85 % de la fenêtre)
   Incluse par défaut dans la stack de `create_deep_agent`. L'historique ancien
   est résumé par le LLM et l'original conservé sur le backend.

5. **Skills en *progressive disclosure***
   Seul le frontmatter (name + description) des 14 `SKILL.md` est chargé au
   démarrage ; le corps n'est lu que si la tâche l'exige.

6. **Memory minimale + garde-taille** (`_MAX_MEMORY_BYTES = 50 000`)
   Les `.md` de `memories/` sont injectés dans le system prompt sous plafond.
   Pour la mémoire volumineuse/cross-session, préférer `CompositeBackend` +
   `StoreBackend` (route `/memories/` vers un store persistant).

7. **Descriptions d'outils concises** — moins de tokens de schéma par outil.

---

## 5. Activation & configuration

```bash
# .env
AGENT_TOPOLOGY=nested   # défaut ; "flat" restaure l'ancienne archi à plat
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-5-20250929
```

Le choix se fait dans `app/agents/master_agent.py::build_master_agent` :

- `nested` → `app.agents.teams.build_nested_subagents(...)` compile 4 chefs
  d'équipe et met `master_tools = []`. Si deepagents est absent, **repli
  automatique** sur `flat`.
- `flat` → comportement historique (`build_subagents`, master avec tous les outils).

Le mode `mock` (pipeline déterministe) est **inchangé** : la topologie n'affecte
que le chemin LLM réel.

Mesurer soi-même :

```python
from app.agents.tools import build_langchain_tools
from app.agents.teams import estimate_context_footprint
print(estimate_context_footprint(build_langchain_tools()))
# {'flat_master_tokens': 2484, 'nested_master_tokens': 218,
#  'master_reduction_pct': 91.2, ...}
```

---

## 6. Compromis & limites

| Bénéfice | Coût / limite |
|----------|---------------|
| Contexte master ~91 % plus léger par tour | +1 niveau d'indirection → latence (un `task()` de plus par équipe) |
| Isolation stricte (pas de fuite d'outputs intermédiaires) | Le master a un contrôle plus grossier de l'ordre ; le HITL fin est délégué **dans** les équipes |
| Slices d'outils par domaine (moins d'erreurs de sélection d'outil) | Un outil partagé (`write_audit_event_tool`) est dupliqué dans plusieurs équipes |
| Repli automatique si deepagents absent | Chaque chef d'équipe est un LLM supplémentaire à orchestrer (coût output, pas input) |

**Quand rester plat ?** Si le workflow tient en peu de tours et que le master
doit garder un contrôle fin de l'ordonnancement + du HITL, la topologie plate
reste plus simple. La topologie imbriquée gagne dès que l'orchestration est
longue et que les sorties d'outils sont volumineuses.

---

## 7. Sources

- [Subagents — docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/subagents)
- [CompiledSubAgent — reference.langchain.com](https://reference.langchain.com/python/deepagents/middleware/subagents/CompiledSubAgent)
- [create_deep_agent — reference.langchain.com](https://reference.langchain.com/python/deepagents/graph/create_deep_agent)
- [Context engineering in Deep Agents — docs.langchain.com](https://docs.langchain.com/oss/python/deepagents/context-engineering)
- [Context Management for Deep Agents — blog LangChain](https://www.langchain.com/blog/context-management-for-deepagents)
- [deepagents (GitHub) — langchain-ai/deepagents](https://github.com/langchain-ai/deepagents)

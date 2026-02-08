# Two-Sided Inventory Playbook

Doc operationnelle de la strategie `run_two_sided_inventory.py`.

## 1) Objectif

Capturer des micro-inefficiences sur Polymarket sans attendre la resolution finale:

- entree sur underpricing (`under_fair`),
- sorties sur `over_fair`, `pair_arb_exit`, `max_hold`, `inv_cap`,
- gestion active de l'inventaire (pas juste "j'achete et j'attends le resultat").

## 2) Ce que fait le bot

Le runner:

- scanne les marches actifs (`gamma-api`),
- lit les books CLOB (`/book`),
- calcule un fair interne (et optionnellement externe via Odds API),
- genere des intents BUY/SELL,
- applique les fills (paper ou live),
- persiste en DB pour dashboard/statistiques.

## 3) Scripts de prod

### 3.1 Two-sided daemon

Fichier: `run_two_sided.sh`

Arguments:

1. `min_edge` (defaut `0.02`)
2. `exit_edge` (defaut `0.006`)
3. `strategy_tag` (auto si non fourni)
4. `mode`: `paper` (defaut) ou `live`
5. `wallet_usd` (defaut `200`)

Exemple paper avec wallet 200:

```bash
/bin/bash /home/ploi/orb.lvlup-dev.com/run_two_sided.sh 0.02 0.006 edge_2p0_0p6_200 paper 200
```

Exemple live avec wallet 200:

```bash
/bin/bash /home/ploi/orb.lvlup-dev.com/run_two_sided.sh 0.02 0.006 edge_2p0_0p6_200 live 200
```

### 3.2 Dashboard daemon

Fichier: `run_dashboard.sh`

```bash
/bin/bash /home/ploi/orb.lvlup-dev.com/run_dashboard.sh
```

## 4) Sizing dynamique (wallet-based)

`run_two_sided.sh` derive automatiquement:

- `min_order = 2.5% wallet`
- `max_order = 4.0% wallet`
- `max_outcome_inv = 12.5% wallet` (min `2 * max_order`)
- `max_market_net = 6.0% wallet` (min `1.2 * max_order`)
- `max_orders_per_cycle = 1` (conservateur)

Avec wallet 200, ordre de grandeur:

- min order ~ `$5`
- max order ~ `$8`
- max outcome inv ~ `$25`
- max market net ~ `$12`

## 5) Dashboard: comment lire les chiffres

Le dashboard montre:

- `Two-Sided Realized`
- `Two-Sided Unrealized` (estime sur mark stocke)
- `Two-Sided Total = realized + unrealized`
- `Two-Sided P&L By Pair` (top/worst par `condition_id`)
- `Open inventory by outcome`

Important:

- un `realized` positif peut masquer un `unrealized` tres negatif.
- toujours regarder `Total` + tableau d'inventaire ouvert.

## 6) Strategy tags (obligatoire)

Toujours lancer avec un `strategy_tag` explicite:

- ex: `edge_2p0_0p6_200`
- ex: `edge_1p5_0p3_200`

Le tag sert a:

- isoler les experiences,
- filtrer l'interface,
- comparer proprement les edges.

## 7) Tests d'edges recommandes

Ne pas lancer trop de variantes en meme temps. Commencer par 2-3:

1. `0.012 / 0.002` (agressif)
2. `0.015 / 0.003` (baseline)
3. `0.020 / 0.006` (conservateur)

Regle de comparaison:

- meme wallet,
- meme fenetre temporelle,
- meme filtres de marche.

## 8) Odds API et credits

Le runner utilise un cache partage en DB (`odds_api_cache`) pour eviter les appels dupliques entre daemons.

Params utiles:

- `--odds-shared-cache` (active)
- `--odds-shared-cache-ttl-seconds 900` (exemple)

Donc plusieurs daemons peuvent reutiliser le meme snapshot au lieu de reconsommer des credits a chaque cycle.

## 9) Reset propre de l'historique

Backup:

```bash
cp data/arb.db "data/arb_backup_$(date +%Y%m%d_%H%M%S).db"
```

Reset:

```bash
sqlite3 data/arb.db "
BEGIN;
DELETE FROM paper_trades;
DELETE FROM live_observations;
DELETE FROM odds_api_cache;
COMMIT;
VACUUM;
"
```

Verification:

```bash
sqlite3 data/arb.db "SELECT COUNT(*) FROM paper_trades;"
sqlite3 data/arb.db "SELECT COUNT(*) FROM live_observations;"
sqlite3 data/arb.db "SELECT COUNT(*) FROM odds_api_cache;"
```

## 10) Checklist quotidienne

1. verifier daemon up (`ps` + logs),
2. verifier que le tag attendu est bien utilise,
3. verifier `Realized / Unrealized / Total` dans dashboard,
4. verifier inventaire ouvert (pas de concentration anormale),
5. verifier credits Odds API restants,
6. ajuster seulement un parametre a la fois.

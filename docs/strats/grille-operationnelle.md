# Grille opérationnelle Polymarket (v1)

Objectif: transformer des idées de stratégies en exécution mesurable, avec contrôle du risque.

## 1) Univers et priorités

Priorité A (le plus exploitable): marchés à résolution claire + liquidité correcte.

Priorité B: marchés "NO" sur scénarios extrêmes, uniquement avec filtres stricts.

Priorité C: marchés 15 minutes ultra-rapides, seulement si exécution semi-auto/auto.

## 2) Score pré-trade (0 à 100)

Ne trader que si score >= 70.

- Edge informationnel (0-30): as-tu une info/process que le marché price mal ?
- Faisabilité exécution (0-25): spread, profondeur, capacité à entrer/sortir proprement.
- Clarté de résolution (0-20): règles de marché non ambiguës.
- Liquidité (0-15): volume + profondeur suffisants pour ta taille.
- Risque de gap/news (0-10): faible risque de saut brutal contre toi.

## 3) Conditions d’entrée (hard rules)

Toutes obligatoires:

1. Coût total estimé (fees + slippage + spread) <= 40% de l’edge brut.
2. Taille max par trade = 0.5% à 1.0% du capital.
3. Exposition max par thème (sport, politics, crypto) <= 20% du capital.
4. Exposition totale simultanée <= 35% du capital.
5. Pas d’entrée si marché/résolution ambiguë.

## 4) Playbooks par type de stratégie

## 4.1 "NO mispricing" (inspiré 4.txt)

Entrée:

1. Scénario objectivement faible probabilité.
2. Prix NO offre un rendement asymétrique raisonnable.
3. Aucun catalyseur court terme capable d’invalider la thèse.

Sortie:

1. Prendre profit partiel sur compression rapide.
2. Sortie totale si nouvelle info augmente la probabilité de l’événement.
3. Stop temporel: si la thèse ne se matérialise pas dans la fenêtre prévue, réduire.

## 4.2 "Research-first" (inspiré 3.txt)

Entrée:

1. Marché shortlisté par scan.
2. Dossier rapide: arguments pour/contre + source primaire.
3. Déséquilibre clair entre narration publique et probabilité implicite.

Sortie:

1. Repricing vers la fair value.
2. Invalidateur fondamental (news contraire).
3. Diminution de liquidité qui empêche une sortie propre.

## 4.3 "Latency / 15m" (inspiré 9.txt)

Règle stricte:

1. En manuel pur: éviter (tu seras souvent en retard).
2. En semi-auto/auto: only si latence et exécution sont testées.

## 5) Gestion du risque

1. Stop journalier: -3R (arrêt de la journée).
2. Stop hebdo: -8R (pause et audit obligatoire).
3. Max 3 pertes consécutives sur un même playbook avant désactivation temporaire.
4. Si drawdown > 10%: réduction de taille de 50%.

## 6) Kill-switch (arrêt immédiat)

1. Changement de règles/fees non intégré.
2. Dégradation exécution (slippage réel > 2x slippage attendu sur 10 trades).
3. Erreur de résolution/règlement détectée.
4. PnL négatif sur 30 trades avec edge théorique positif.

## 7) KPIs hebdomadaires

1. Winrate (utile, mais secondaire).
2. Expectancy par trade (KPI principal).
3. Edge capturé / edge théorique.
4. Slippage moyen réel.
5. Temps moyen de détention.
6. PnL par playbook (NO, research, latency).
7. Max drawdown.

## 8) Routine d’amélioration (chaque semaine)

1. Garder uniquement les 20% de setups qui font 80% du PnL.
2. Supprimer les marchés à résolution litigieuse.
3. Ajuster tailles selon expectancy, pas selon confiance subjective.
4. Mettre à jour checklist d’entrée avec les erreurs récurrentes.

## 9) Règle d’or

Si une stratégie ne survit pas aux fees + slippage + latence en backtest réaliste, elle n’existe pas en réel.


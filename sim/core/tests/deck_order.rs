//! Deck creation order: card_protos sorted by the `s..r` string
//! (game.lua:2367-2369) => suits C<D<H<S, ranks by ASCII
//! '2'..'9','A','J','K','Q','T'.

use balatro_core::cards::Suit;
use balatro_core::deck::red_deck;

#[test]
fn creation_order_and_sort_ids() {
    let deck = red_deck();
    assert_eq!(deck.len(), 52);
    // sort_ids are 1..=52 in creation order.
    for (i, card) in deck.iter().enumerate() {
        assert_eq!(card.sort_id, i as u32 + 1);
    }

    let key = |i: usize| format!("{}_{}", deck[i].suit.key(), deck[i].rank.key());
    // Clubs block: 2..9, A, J, K, Q, T.
    let club_ranks: Vec<char> = (0..13).map(|i| deck[i].rank.key()).collect();
    assert_eq!(
        club_ranks,
        vec!['2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q', 'T']
    );
    assert!(deck[..13].iter().all(|c| c.suit == Suit::Clubs));
    assert!(deck[13..26].iter().all(|c| c.suit == Suit::Diamonds));
    assert!(deck[26..39].iter().all(|c| c.suit == Suit::Hearts));
    assert!(deck[39..].iter().all(|c| c.suit == Suit::Spades));

    assert_eq!(key(0), "C_2");
    assert_eq!(key(8), "C_A");
    assert_eq!(key(12), "C_T");
    assert_eq!(key(13), "D_2");
    assert_eq!(key(51), "S_T");

    // Every rank+suit combo appears exactly once.
    let mut seen = std::collections::HashSet::new();
    for card in &deck {
        assert!(seen.insert((card.rank.id(), card.suit)));
    }
}

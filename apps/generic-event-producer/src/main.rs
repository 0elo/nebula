#[macro_use]
extern crate maplit;

mod notifier;

fn main() {
    let generic_event = notifier::GenericEvent {
        name: String::from("Test Event"),
        payload: hashmap! {
            String::from("k1") => String::from("v1"),
            String::from("foo") => String::from("bar"),
        },
        metadata: notifier::KafkaMetadata {
            topic: String::from("generic-event-1"),
        },
    };

    let event_notifier: Box<dyn notifier::GenericEventNotifier<notifier::KafkaMetadata>> =
        Box::new(notifier::GenericEventNotifierKafka::new());

    event_notifier.notify(&generic_event);
}
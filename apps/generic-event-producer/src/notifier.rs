use std::collections::HashMap;

pub struct GenericEvent<T> {
    pub name: String,
    pub payload: HashMap<String, String>,
    pub metadata: T,
}

pub trait GenericEventNotifier<T> {
    fn notify(&self, event: &GenericEvent<T>);
}

pub struct KafkaMetadata {
    pub topic: String,
}

pub struct GenericEventNotifierKafka {}
impl GenericEventNotifierKafka {
    pub fn new() -> Self {
        GenericEventNotifierKafka {}
    }
}

impl GenericEventNotifier<KafkaMetadata> for GenericEventNotifierKafka {
    fn notify(&self, event: &GenericEvent<KafkaMetadata>) {
        println!(
            "Producing event with name: {}!\nPayload is: {:?}\nTopic name is {}",
            event.name, event.payload, event.metadata.topic
        );
    }
}
#include "mqtt.h"

mqtt::mqtt(const char * id, const char * application_topic, const char * broker, int port) : mosquittopp(id) {
    mosqpp::lib_init();
    this->keepalive = DEFAULT_KEEP_ALIVE;
    this->id = id;
    this->port = port;
    this->broker = broker;
    std::string base_topic = GENERAL_TOPIC;
    std::string app_topic = application_topic;
    std::string sender_topic = id;
    topic = base_topic +  '/' + app_topic + '/' + sender_topic;
    this->topic = topic;
    connect_async(broker, port, keepalive);
    loop_start();
}

mqtt::~mqtt() {
    loop_stop();
    mosqpp::lib_cleanup();
}

bool mqtt::send_message(const char * message) {
     // Send message - depending on QoS, mosquitto lib managed re-submission this the thread
     //
     // * NULL : Message Id (int *) this allow to latter get status of each message
     // * topic : topic to be used
     // * lenght of the message
     // * message
     // * qos (0,1,2)
     // * retain (boolean) - indicates if message is retained on broker or not
     // Should return MOSQ_ERR_SUCCESS
     int ret = publish(NULL, this->topic.c_str(), strlen(message), message, 1, false);
     return ( ret == MOSQ_ERR_SUCCESS );
 }

 void mqtt::on_disconnect(int rc) {
     std::cout << ">> mqtt - disconnection(" << rc << ")" << std::endl;
 }

void mqtt::on_connect(int rc) {
     if ( rc == 0 ) {
     std::cout << ">> mqtt - connected with server" << std::endl;
     } else {
     std::cout << ">> mqtt - Impossible to connect with server(" << rc << ")" << std::endl;
     }
 }

void mqtt::on_publish(int mid)
{
    std::cout << ">> mqtt - Message (" << mid << ") succeed to be published " << std::endl;
    //std::cout << this->topic << std::endl;
}

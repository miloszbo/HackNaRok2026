#include <zephyr/kernel.h>
#include <zephyr/device.h>
#include <zephyr/drivers/sensor.h>

int main(void)
{
    // Pobranie konfiguracji czujnika z pliku .overlay (po nazwie węzła 'hcsr04_sensor')
    const struct device *const hcsr04 = DEVICE_DT_GET(DT_NODELABEL(hcsr04_sensor));

    // Sprawdzenie, czy system poprawnie zainicjował czujnik
    if (!device_is_ready(hcsr04)) {
        printk("Blad: Czujnik HC-SR04 nie jest gotowy do pracy!\n");
        return 0;
    }

    printk("Rozpoczynam pomiary na nRF54L15 (logika 3.3V)...\n");

    while (1) {
        struct sensor_value distance;

        // 1. Wyslanie impulsu i zebranie pomiaru z czujnika
        int ret = sensor_sample_fetch(hcsr04);
        
        if (ret == 0) {
            // 2. Wyciągnięcie konkretnej wartości (odległości)
            sensor_channel_get(hcsr04, SENSOR_CHAN_DISTANCE, &distance);
            
            // Zephyr zwraca odległość w metrach w dwóch zmiennych:
            // val1 to metry (liczba całkowita), val2 to milionowe części metra (mikrometry)
            printk("Odleglosc: %d.%06d m\n", distance.val1, distance.val2);
        } else {
            printk("Blad odczytu z czujnika. Kod bledu: %d\n", ret);
        }

        // Czekaj 1 sekundę (1000 milisekund) przed kolejnym strzałem
        k_msleep(1000);
    }
    return 0;
}

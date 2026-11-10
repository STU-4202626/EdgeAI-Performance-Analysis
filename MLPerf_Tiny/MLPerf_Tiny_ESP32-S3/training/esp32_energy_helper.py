# ESP32-S3 energy and carbon estimation helper.
#
# Power estimate source:
# Espressif Systems, ESP32-S3 Series Datasheet v2.2, Section 5.6.2, Table 5-9: Current Consumption in Modem-sleep Mode.
# https://www.espressif.com/documentation/esp32-s3_datasheet_en.pdf
#
# At 240 MHz, Espressif reports a typical current of 87.9 mA for single-core 128-bit data access instructions with the other core idle when peripheral clocks are enabled.
#
# The datasheet current is used as an estimate rather than a direct electrical measurement of the development board. Actual consumption may differ depending on peripherals, memory and board configuration.

ESP32_VOLTAGE_V = 3.3
ESP32_CURRENT_A = 0.0879

ESP32_ESTIMATED_POWER_W = (
    ESP32_VOLTAGE_V * ESP32_CURRENT_A
)

# Carbon intensity derived from the existing CodeCarbon benchmark results recorded for South Africa, Western Cape.
# Both the CPU and Raspberry Pi 5 CodeCarbon results used: 0.707686 kgCO2e/kWh.
CARBON_INTENSITY_KG_PER_KWH = 0.707686


def estimate_esp32_energy_and_emissions(inference_time_s):

    if inference_time_s < 0:
        raise ValueError("Inference time cannot be negative.")

    # Convert estimated ESP32-S3 power and inference time into kWh.
    energy_kwh = (
        ESP32_ESTIMATED_POWER_W * inference_time_s / 3_600_000.0
    )

    # Estimate carbon emissions using the same carbon intensity applied to the other benchmark devices.
    emissions_kg = (
        energy_kwh * CARBON_INTENSITY_KG_PER_KWH
    )

    return energy_kwh, emissions_kg
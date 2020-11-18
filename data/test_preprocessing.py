import unittest
from textwrap import dedent

from preprocessing import DocumentPreprocessor

TEST_SAMPLES = [
    {
        'description': 'Missing Paragraph breaks (Notes)',
        'raw_text': """
Notes to the consolidated financial statements – Part B – Information on the consolidated balance sheet - Assets 


time in 2018 should continue to blunt the upwards pressure exerted by the movements of the U.S. dollar curve. Nonetheless, 
the probable end of the ECB’s purchase programme by the end of 2018, and thus the growing expectations of increases in 
official interest rates, will lead to an acceleration in medium and long-term interest rates. Finally, the scenario assumes a 
further increase in risk premiums on Italian debt in 2018, following that already seen from 2016 to 2017. The causes are 
linked to the mounting political instability, the retraction of the protective net extended by monetary authorities in recent years 
and the slow pace of the process of reducing debt. The reference scenario predicts an increase in the ten-year BTP-Bund 
spread to 190-210 bps, without significant declines in 2019. Only in 2020-22, when it is predicted that a virtuous scenario of 
accelerating growth and falling debt will occur, in an environment of continuing low interest rates, will the spread fall towards 
150 and then 100 bps. 
With reference to the banking industry, in 2017 loans to the private sector confirmed moderate growth, once again sustained 
by loans to households, both for mortgage loans and consumer credit. However, mortgage lending, despite remaining at high 
volumes, was down year-on-year, mainly due to the decrease in renegotiations, although new agreements also declined. In 
conditions of liquidity considered sufficient or more than sufficient by most companies, demand for credit remained subdued 
by low external funding requirements, which were also met by bond issues. Business debt as a percentage of GDP continued 
to fall, with loans to businesses only starting to pick up towards the end of the year.  
Also in 2018 and over the entire time horizon, monetary conditions will remain favourable to credit recovery. Lagging 
significantly behind the economic recovery, a clear resumption of growth of loans to businesses should be seen in 2018. 
However, the increase in loans to businesses will continue to be restrained due to demand factors, such as the presumable 
continuation of corporate deleveraging and ongoing diversification in favour of market sources of funding, in a scenario of 
moderate GDP growth.  
For households, the lending scenario remains positive. Loan levels will continue to rise in 2018-22, driven by the positive 
outlook for the real-estate market and interest rates that will remain near all-time lows in the next few years. 
As regards bank funding, the trends seen in 2017 confirmed the previous values, and specifically the growth in deposits and 
the decline for bonds. With interest rates starting to rise in 2020, households and businesses will focus on instruments with 
higher yields, to the partial disadvantage of more liquid instruments. This trend will be accompanied by offering policies, with 
banks that must handle increased needs for funding given the consolidation of the growth in loans and the gradual elimination 
of non-conventional operations of the ECB. Overall, deposits are expected to grow over the entire forecast period. 
For bonds, over the entire time horizon of the scenario, net redemptions of bonds placed in the retail segment are assumed. 
On the other hand, especially near the end of the forecast period, the drop-in stock could be counterbalanced by issues on 
the wholesale market, fuelled by the placement of eligible instruments that meet the requirements of loss absorption, in a 
scenario of gradual elimination of non-conventional monetary stimulus and moderate acceleration of medium/long-term loans. 
Customer deposits will recover, gradually showing very moderate growth, also due to the funding requirements linked to the 
redemption of the TLTRO in 2020-21.  
In 2017, bank interest rates were still dropping, despite the attempts to increase mortgage lending rates in the first part of the 
year and rates applied to businesses on large loans having settled. At the same time, interest rates offered to businesses on 
smaller loans reached new lows. The spread between lending and funding rates fell slightly. For the sixth consecutive year, in 
2017 the mark-down on demand deposits was in negative territory, where it is expected to remain until 2021, given the very 
low Euribor rates. In particular, in 2018 and for the average of 2019, the mark-down is expected to remain essentially 
unchanged. A trend towards a gradual improvement is expected to begin at the end of 2019, followed by stabilisation, and 
continue in 2021-22 in line with the presumed profile of the Euribor. The mark-up on short-term interest rates in 2017 
decreased further. In 2018 it is expected to slightly decline further in the comparison of annual averages, but to remain 
essentially stable compared to the end of 2017 and higher than pre-crisis levels, given the low levels of monetary rates. The 
reduction will resume at the end of 2019 and the end of 2021 as market rates begin to rise once more.  
Lastly, in relation to the asset management sector (mutual funds, portfolio management schemes and insurance reserves) in 
2017 mutual funds showed recovery, fuelled by the accumulation of liquidity that occurred in recent years in financial assets 
of households, and supported by the improved performance of the markets compared to 2016. In 2018-19, the scenario of 
continued very low interest rates will favour the steadiness of this segment, though with lower flows than in 2017, due to the 
gradual exhaustion of the margins for shifting portfolios of households, which will continue over the entire time horizon of the 
scenario. Finally, in 2018 market performance expectations remain cautious, especially due to the uncertainty of the political 
scenario, a factor that restrains volume growth. 
In the subsequent years, mutual funds inflows are expected to continue to record positive, but gradually decreasing values, as 
a result of the consolidation of the recovery in deposits from banking business and the forecast increase in interest rates. For 
life insurance, premiums will see moderate growth, given the high volumes already achieved by the segment, supported in the 
last few years of the scenario by the increase in interest rates, which will breathe new life into traditional insurance. Starting in 
2020, this will also favour flows routed to portfolio management schemes, supported by the institutional component. Finally, 
due to the combined effect of positive inflows and revaluations linked to the stock market, the total assets of the asset 
management system are expected to continue growing at a good pace. 


264
""",
        'expected_text': """time in 2018 should continue to blunt the upwards pressure exerted by the movements of the U.S. dollar curve. Nonetheless, 
the probable end of the ECB’s purchase programme by the end of 2018, and thus the growing expectations of increases in 
official interest rates, will lead to an acceleration in medium and long-term interest rates. Finally, the scenario assumes a 
further increase in risk premiums on Italian debt in 2018, following that already seen from 2016 to 2017. The causes are 
linked to the mounting political instability, the retraction of the protective net extended by monetary authorities in recent years 
and the slow pace of the process of reducing debt. The reference scenario predicts an increase in the ten-year BTP-Bund 
spread to 190-210 bps, without significant declines in 2019. Only in 2020-22, when it is predicted that a virtuous scenario of 
accelerating growth and falling debt will occur, in an environment of continuing low interest rates, will the spread fall towards 
150 and then 100 bps. 
With reference to the banking industry, in 2017 loans to the private sector confirmed moderate growth, once again sustained 
by loans to households, both for mortgage loans and consumer credit. However, mortgage lending, despite remaining at high 
volumes, was down year-on-year, mainly due to the decrease in renegotiations, although new agreements also declined. In 
conditions of liquidity considered sufficient or more than sufficient by most companies, demand for credit remained subdued 
by low external funding requirements, which were also met by bond issues. Business debt as a percentage of GDP continued 
to fall, with loans to businesses only starting to pick up towards the end of the year. 
Also in 2018 and over the entire time horizon, monetary conditions will remain favourable to credit recovery. Lagging 
significantly behind the economic recovery, a clear resumption of growth of loans to businesses should be seen in 2018. 
However, the increase in loans to businesses will continue to be restrained due to demand factors, such as the presumable 
continuation of corporate deleveraging and ongoing diversification in favour of market sources of funding, in a scenario of 
moderate GDP growth. 
For households, the lending scenario remains positive. Loan levels will continue to rise in 2018-22, driven by the positive 
outlook for the real-estate market and interest rates that will remain near all-time lows in the next few years. 
As regards bank funding, the trends seen in 2017 confirmed the previous values, and specifically the growth in deposits and 
the decline for bonds. With interest rates starting to rise in 2020, households and businesses will focus on instruments with 
higher yields, to the partial disadvantage of more liquid instruments. This trend will be accompanied by offering policies, with 
banks that must handle increased needs for funding given the consolidation of the growth in loans and the gradual elimination 
of non-conventional operations of the ECB. Overall, deposits are expected to grow over the entire forecast period. 
For bonds, over the entire time horizon of the scenario, net redemptions of bonds placed in the retail segment are assumed. 
On the other hand, especially near the end of the forecast period, the drop-in stock could be counterbalanced by issues on 
the wholesale market, fuelled by the placement of eligible instruments that meet the requirements of loss absorption, in a 
scenario of gradual elimination of non-conventional monetary stimulus and moderate acceleration of medium/long-term loans. 
Customer deposits will recover, gradually showing very moderate growth, also due to the funding requirements linked to the 
redemption of the TLTRO in 2020-21. 
In 2017, bank interest rates were still dropping, despite the attempts to increase mortgage lending rates in the first part of the 
year and rates applied to businesses on large loans having settled. At the same time, interest rates offered to businesses on 
smaller loans reached new lows. The spread between lending and funding rates fell slightly. For the sixth consecutive year, in 
2017 the mark-down on demand deposits was in negative territory, where it is expected to remain until 2021, given the very 
low Euribor rates. In particular, in 2018 and for the average of 2019, the mark-down is expected to remain essentially 
unchanged. A trend towards a gradual improvement is expected to begin at the end of 2019, followed by stabilisation, and 
continue in 2021-22 in line with the presumed profile of the Euribor. The mark-up on short-term interest rates in 2017 
decreased further. In 2018 it is expected to slightly decline further in the comparison of annual averages, but to remain 
essentially stable compared to the end of 2017 and higher than pre-crisis levels, given the low levels of monetary rates. The 
reduction will resume at the end of 2019 and the end of 2021 as market rates begin to rise once more. 
Lastly, in relation to the asset management sector (mutual funds, portfolio management schemes and insurance reserves) in 
2017 mutual funds showed recovery, fuelled by the accumulation of liquidity that occurred in recent years in financial assets 
of households, and supported by the improved performance of the markets compared to 2016. In 2018-19, the scenario of 
continued very low interest rates will favour the steadiness of this segment, though with lower flows than in 2017, due to the 
gradual exhaustion of the margins for shifting portfolios of households, which will continue over the entire time horizon of the 
scenario. Finally, in 2018 market performance expectations remain cautious, especially due to the uncertainty of the political 
scenario, a factor that restrains volume growth. 
In the subsequent years, mutual funds inflows are expected to continue to record positive, but gradually decreasing values, as 
a result of the consolidation of the recovery in deposits from banking business and the forecast increase in interest rates. For 
life insurance, premiums will see moderate growth, given the high volumes already achieved by the segment, supported in the 
last few years of the scenario by the increase in interest rates, which will breathe new life into traditional insurance. Starting in 
2020, this will also favour flows routed to portfolio management schemes, supported by the institutional component. Finally, 
due to the combined effect of positive inflows and revaluations linked to the stock market, the total assets of the asset 
management system are expected to continue growing at a good pace. """
    },
    {
        'description': "Paragraph breaks",
        'raw_text': """
Almost a quarter of flaring in our Upstream and Integrated Gas facilities in 
2017 took place in assets operated by The Shell Petroleum Development 
Company of Nigeria Limited (SPDC). Flaring from SPDC-operated facilities fell 
by more than 40% between 2013 and 2017. However, flaring intensity 
levels in SPDC increased in 2017 compared with 2016, partly due to the 
restart of facilities that were offline for most of 2016. Several new gas-


gathering projects came on stream at the end of 2017. However, the 
planned start-up dates for two gas-gathering projects have historically been 
delayed due to lack of adequate joint venture funding. Nevertheless, with 
funding now restored, the projects are planned for completion in 2018-19.


GHG emissions data are provided below in accordance with UK regulations. 
GHG emissions comprise CO2, methane, nitrous oxide, hydrofluorocarbons, 
perfluorocarbons, sulphur hexafluoride and nitrogen trifluoride. The data are 
calculated using locally regulated methods where they exist. Where there is 
no locally regulated method, the data are calculated using the 2009 API 
Compendium, which is the recognised industry standard under the GHG 
Protocol Corporate Accounting and Reporting Standard. There are inherent 
limitations to the accuracy of such data. Oil and gas industry guidelines 
(IPIECA/API/IOGP) indicate that a number of sources of uncertainty can 
contribute to the overall uncertainty of a corporate emissions inventory. 
""",
        'expected_text': """
Almost a quarter of flaring in our Upstream and Integrated Gas facilities in 
2017 took place in assets operated by The Shell Petroleum Development 
Company of Nigeria Limited (SPDC). Flaring from SPDC-operated facilities fell 
by more than 40% between 2013 and 2017. However, flaring intensity 
levels in SPDC increased in 2017 compared with 2016, partly due to the 
restart of facilities that were offline for most of 2016. Several new gasgathering 
projects came on stream at the end of 2017. However, the 
planned start-up dates for two gas-gathering projects have historically been 
delayed due to lack of adequate joint venture funding. Nevertheless, with 
funding now restored, the projects are planned for completion in 2018-19.

GHG emissions data are provided below in accordance with UK regulations. 
GHG emissions comprise CO2, methane, nitrous oxide, hydrofluorocarbons, 
perfluorocarbons, sulphur hexafluoride and nitrogen trifluoride. The data are 
calculated using locally regulated methods where they exist. Where there is 
no locally regulated method, the data are calculated using the 2009 API 
Compendium, which is the recognised industry standard under the GHG 
Protocol Corporate Accounting and Reporting Standard. There are inherent 
limitations to the accuracy of such data. Oil and gas industry guidelines 
(IPIECA/API/IOGP) indicate that a number of sources of uncertainty can 
contribute to the overall uncertainty of a corporate emissions inventory. 
"""
    },
    {
        'description': "Test Tables",
        'raw_text': """
This is an introduction.

Oil and gas acreage (at December 31) Thousand acres
   2017 2016   2015
  
 Developed   Undeveloped Developed Undeveloped   Developed   Undeveloped
   Gross   Net   Gross Net Gross Net Gross Net   Gross   Net   Gross Net

Europe [A]   6,463    2,071    14,119 6,187 6,556 2,197 18,216 10,241    7,152    2,194    14,623 7,732
Asia   25,975    9,139    35,305 18,730 26,003 9,199 58,463 36,298    25,581    9,181    36,658 22,995
Oceania   3,296    1,255    22,406 13,985 1,939 822 37,876 24,109    2,041    530    51,740 16,975
Africa   4,663    1,938    33,453 20,811 5,083 2,315 41,517 29,152    4,650    2,071    40,435 27,058
North America – USA   1,936    1,134    2,718 1,937 2,002 1,197 4,151 2,577    1,659    1,158    5,033 4,262
North America – Canada   953    651    16,714 15,005 976 670 26,149 19,402    1,227    745    32,706 25,716
South America   1,302    606    9,338 6,196 1,315 547 17,759 14,643    100    52    7,851 3,621
Total   44,588    16,794    134,053 82,851 43,874 16,947 204,131 136,422    42,410    15,931    189,046 108,359  
[A] Includes Greenland. 
""",
        'expected_text': """
This is an introduction.

Oil and gas acreage (at December 31) Thousand acres

Developed Undeveloped Developed Undeveloped Developed Undeveloped
Gross Net Gross Net Gross Net Gross Net Gross Net Gross Net

[A] Includes Greenland. 
"""
    }, {
        'description': 'Stripping',
        'raw_text': """
        Fight climate change and improve the efficiency and use of resources.
Promote the use of affordable, safe, sustainable and modern energy for all.
Ensure sustainable consumption and production models.


 » Reduction 50% GHG 
(Scope 1 + 2) by 2030 
(year base 2015).


 » Reduce +70% energy 
consumption per unit 
of traffic by 2020.


 » Achieve 100% 
of electricity 
consumption 
from renewable 
sources by 2030.


 » Reduce 30% 
emissions of 
suppliers/€ destined 
to purchase by 2025 
(year base 2016).


 » Boost waste 
recycling.
        """,
        'expected_text': """
Fight climate change and improve the efficiency and use of resources.
Promote the use of affordable, safe, sustainable and modern energy for all.
Ensure sustainable consumption and production models.


 » Reduction 50% GHG 
(Scope 1 + 2) by 2030 
(year base 2015).


 » Reduce +70% energy 
consumption per unit 
of traffic by 2020.


 » Achieve 100% 
of electricity 
consumption 
from renewable 
sources by 2030.


 » Reduce 30% 
emissions of 
suppliers/€ destined 
to purchase by 2025 
(year base 2016).


 » Boost waste 
recycling."""
    }
]


class PreprocessingTestCase(unittest.TestCase):
    """Test suite that contains small unit tests and "integration tests" for the text preprocessing
    """

    def test__all_samples(self):
        """Tests all provided samples by using sub-tests"""
        for s in TEST_SAMPLES:
            with self.subTest(msg=s['description']):
                result = DocumentPreprocessor(doc=s['raw_text']).process()
                self.assertEqual(result, s['expected_text'])

    def test_co2(self):
        """Tests if the case of co2 that split a line is processed properly"""

        text = dedent("""
        Regarding climate change, CO
        2
        is the main contributor
        """)
        correct_text = dedent("""
        Regarding climate change, CO2 is the main contributor
        """)

        result = DocumentPreprocessor(doc=text).process()
        self.assertEqual(result, correct_text)

    def test_number(self):

        text = dedent("""
        Local national coverage (at December 31) 
        Number of selected key business countries.


        2017    2016 2015
        Greater than 80%    10       10    12
        Less than 80%    10       10    8
        Total    20       20    20
        
        CODE OF CONDUCT 
        In line with the UN Global Compact Principle 10 (Businesses should work 
        against corruption in all its forms, including extortion and bribery), we maintain 
        a global Anti Bribery and Corruption (ABC) programme designed to prevent or 
        detect, and remediate and learn from, potential violations. The programme is 
        """)

        correct_text = dedent("""
        Local national coverage (at December 31) 
        Number of selected key business countries.
        
        CODE OF CONDUCT 
        In line with the UN Global Compact Principle 10 (Businesses should work 
        against corruption in all its forms, including extortion and bribery), we maintain 
        a global Anti Bribery and Corruption (ABC) programme designed to prevent or 
        detect, and remediate and learn from, potential violations. The programme is 
        """)
        result = DocumentPreprocessor(
            text).process()
        self.assertEqual(result, correct_text)


if __name__ == '__main__':
    unittest.main()

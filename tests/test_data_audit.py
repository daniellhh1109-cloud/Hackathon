import unittest
import pandas as pd
from src.data.data_audit import history_lengths

class CalendarHistoryTests(unittest.TestCase):
    def test_gap_is_not_twelve_months(self):
        dates=pd.date_range('2020-01-31',periods=13,freq='ME').delete(5)
        out=history_lengths(pd.DataFrame({'permno':1,'eom':dates}))
        self.assertFalse(out.has_12_calendar_months.any())
    def test_unsorted_duplicates_and_security_boundary(self):
        dates=list(pd.date_range('2020-01-31',periods=12,freq='ME'))
        x=pd.DataFrame({'permno':[1]*13+[2],'eom':dates+[dates[0],dates[-1]]}).sample(frac=1,random_state=1)
        out=history_lengths(x)
        self.assertEqual(int(out.has_12_calendar_months.sum()),1)
        self.assertEqual(int(out.loc[out.permno.eq(2),'consecutive_months'].iloc[0]),1)

if __name__=='__main__': unittest.main()

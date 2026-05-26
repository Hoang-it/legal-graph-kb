years_contributed(user, 1).
legal_source(source_c051, law_bhxh_2014, article_26, clause_1, none, 'Đã đóng BHXH dưới 15 năm: tối đa 30 ngày làm việc/năm; đóng đủ 15 đến dưới 30 năm: tối đa 40 ngày làm việc/năm; đóng đủ 30 năm trở lên: tối đa 60 ngày làm việc/năm.').
sick_leave_max_days(Person, Days, Trace) :- years_contributed(Person, Years), Years < 15, Days = 30, Trace = [step(conclusion, sick_leave_max_days(Person, Days), based_on(source_c051))].

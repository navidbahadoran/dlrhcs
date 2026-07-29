series {
  title = "bps_permits 37380"
  start = 2007.1
  period = 12
  data = (
54
49
59
54
48
73
57
65
34
43
27
25
29
19
18
26
29
34
258
93
22
18
13
104
20
13
9
13
11
6
18
14
13
14
11
26
9
19
54
94
12
13
19
17
12
11
12
10
17
10
17
3
13
24
8
21
16
7
11
10
20
16
14
17
23
15
16
23
22
34
21
22
38
31
24
30
40
44
38
47
76
80
63
23
  )
}
transform { function = none }
regression { aictest = (td easter) }
automdl {}
x11 { mode = add save = (d10 d11) }

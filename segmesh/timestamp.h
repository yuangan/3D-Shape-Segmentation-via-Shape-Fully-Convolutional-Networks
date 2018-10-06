#ifndef TIMESTAMP_H
#define TIMESTAMP_H
/*
Szymon Rusinkiewicz
Princeton University

timestamp.h
Wrapper around system-specific timestamps.
*/



# define WIN32_LEAN_AND_MEAN
# include <limits.h>
# include <windows.h>
# define usleep(x) Sleep((x)/1000)

  struct timestamp { LARGE_INTEGER t; };

  static inline double LI2d(const LARGE_INTEGER &li)
  {
	// Work around random compiler bugs...
	double d = *(unsigned *)(&(li.HighPart));
	d *= 65536.0 * 65536.0;
	d += *(unsigned *)(&(li.LowPart));
	return d;
  }

  static inline float operator - (const timestamp &t1, const timestamp &t2)
  {
	static LARGE_INTEGER PerformanceFrequency;
	static int status = QueryPerformanceFrequency(&PerformanceFrequency);
	if (status == 0) return 1.0f;

	return float((LI2d(t1.t) - LI2d(t2.t)) / LI2d(PerformanceFrequency));
  }

  static inline timestamp now()
  {
	timestamp t;
	QueryPerformanceCounter(&t.t);
	return t;
  }

#endif



MODULE := src/predictors

MODULE_OBJS := \
	src/predictors/CNN_AE.o \

MODULE_DIRS += \
	src/predictors

# Include common rules 
include $(srcdir)/common.rules

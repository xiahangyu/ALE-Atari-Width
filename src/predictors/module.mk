MODULE := src/predictors

MODULE_OBJS := \
	src/predictors/model_loader.o \
	src/predictors/cs_predictor.o \
	src/predictors/os_predictor.o \
	src/predictors/ns_predictor.o \

MODULE_DIRS += \
	src/predictors

# Include common rules 
include $(srcdir)/common.rules
